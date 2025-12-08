from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from datetime import datetime
import time
from typing import Dict, List, Any, Optional

from config.settings import settings

DEBUG = __name__ == "__main__"


class FuzzySearchService:
    """Service for fuzzy/vector-based student search operations."""

    def __init__(self, search_endpoint: str, index_name: str, credential):
        self.search_endpoint = search_endpoint
        self.index_name = index_name
        self.credential = credential

        # Fields we actually need back from AI Search
        self._select_fields = [
            "student_id",
            "pen",
            "legalFirstName",
            "legalMiddleNames",
            "legalLastName",
            "dob",
            "sexCode",
            "postalCode",
            "mincode",
            "gradeCode",
            "localID",
        ]

        # OpenAI embedding client
        self.openai_client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version="2023-05-15",
            azure_endpoint=settings.openai_api_base_embedding,
        )

        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=self.credential,
        )

    # ------------------------------------------------------------------
    # Embedding generation (USED ONLY IN FUZZY PATH)
    # ------------------------------------------------------------------
    def generate_embedding(self, query_data: Dict[str, Any]) -> Optional[List[float]]:
        """Generate embedding from full name using Azure OpenAI embedding model."""

        # Prefer legal* fields, fall back to givenName/surname/middleName if present
        first = (
            (query_data.get("legalFirstName")
             or query_data.get("givenName")
             or "")
            .strip()
        )
        last = (
            (query_data.get("legalLastName")
             or query_data.get("surname")
             or "")
            .strip()
        )
        middle = (
            (query_data.get("legalMiddleNames")
             or query_data.get("middleName")
             or "")
            .strip()
        )

        # Normalize "NULL"
        if first == "NULL":
            first = ""
        if last == "NULL":
            last = ""
        if middle == "NULL":
            middle = ""

        parts = [p for p in [first, middle, last] if p]
        text = " ".join(parts) + "."

        if text == ".":
            # no name available → no embedding
            return None

        try:
            if DEBUG:
                print(f"Generating embedding for: {text}")
            t0 = time.perf_counter()
            resp = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
            )
            t1 = time.perf_counter()
            if DEBUG:
                print(f"Embedding generation took {t1 - t0:.3f} seconds")
            return resp.data[0].embedding
        except Exception as e:
            if DEBUG:
                print(f"Error generating embedding for '{text}': {e}")
            return None

    # ------------------------------------------------------------------
    # Helpers: normalization + similarity
    # ------------------------------------------------------------------
    def _normalize_query_dob(self, dob_str: str) -> str:
        """
        Normalize query DOB to 'YYYY-MM-DD'.
        Accepts 'YYYYMMDD' or 'YYYY-MM-DD'. Returns '' if invalid.
        """
        dob_str = (dob_str or "").strip()
        if not dob_str:
            return ""
        try:
            if "-" in dob_str:
                dt = datetime.strptime(dob_str, "%Y-%m-%d")
            else:
                dt = datetime.strptime(dob_str, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return ""

    def _escape_filter_str(self, value: str) -> str:
        """Escape single quotes for OData filter strings."""
        return (value or "").replace("'", "''")

    def _dob_literal(self, dob_iso: str) -> str:
        """
        Turn 'YYYY-MM-DD' into a filter literal.
        If dob is stored as Edm.String in the index, this is enough:
          dob eq '2001-10-02'
        """
        return f"'{dob_iso}'"

    def _postal_similarity(self, query_postal: str, candidate_postal: str) -> float:
        """
        Simple, cheap similarity for Canadian postal codes:
        - Exact normalized match → 1.0
        - Same FSA (first 3 chars) → 0.7
        - Same first 2 chars → 0.5
        - Otherwise → 0.0
        """
        if not query_postal or not candidate_postal:
            return 0.0

        q = query_postal.replace(" ", "").upper()
        c = candidate_postal.replace(" ", "").upper()

        if q == c:
            return 1.0

        # same FSA (first 3 chars)
        if len(q) >= 3 and len(c) >= 3 and q[:3] == c[:3]:
            return 0.7

        # same first 2 chars
        if len(q) >= 2 and len(c) >= 2 and q[:2] == c[:2]:
            return 0.5

        return 0.0

    def _mincode_similarity(self, query_mincode: str, candidate_mincode: str) -> float:
        """
        Simple numeric/string similarity for mincode:
        - Exact match (after strip + leading-zero normalization) → 1.0
        - Same first 4 chars → 0.8
        - Same first 3 chars → 0.6
        - Otherwise → 0.0
        """
        if not query_mincode or not candidate_mincode:
            return 0.0

        q = str(query_mincode).strip()
        c = str(candidate_mincode).strip()

        # normalize leading zeros
        q_norm = q.lstrip("0")
        c_norm = c.lstrip("0")

        if q_norm == c_norm:
            return 1.0

        max_prefix = min(len(q_norm), len(c_norm))

        if max_prefix >= 4 and q_norm[:4] == c_norm[:4]:
            return 0.8
        if max_prefix >= 3 and q_norm[:3] == c_norm[:3]:
            return 0.6

        return 0.0

    def _sex_similarity(self, query_sex: str, candidate_sex: str) -> float:
        if not query_sex or not candidate_sex:
            return 0.0
        return 1.0 if query_sex.upper() == candidate_sex.upper() else 0.0

    def _dob_similarity(self, query_dob: str, candidate_dob: Any) -> float:
        """
        query_dob is 'YYYY-MM-DD' or ''.
        candidate_dob may be datetime or string.
        """
        if not query_dob or not candidate_dob:
            return 0.0

        if hasattr(candidate_dob, "strftime"):
            cand_str = candidate_dob.strftime("%Y-%m-%d")
        else:
            cand_str = str(candidate_dob)[:10]

        return 1.0 if query_dob == cand_str else 0.0

    # ------------------------------------------------------------------
    # Single vector search call with optional filter
    # ------------------------------------------------------------------
    def _vector_search_once(
        self,
        embedding: List[float],
        filter_expr: Optional[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        vector_queries: List[VectorizedQuery] = [
            VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=top_k,
                fields="nameEmbedding",
                exhaustive=False,  # could be True for exact, slower but more recall
            )
        ]

        search_kwargs: Dict[str, Any] = {
            "search_text": "*",
            "vector_queries": vector_queries,
            "top": top_k,
            "include_total_count": False,
            "select": self._select_fields,
        }
        if filter_expr:
            search_kwargs["filter"] = filter_expr

        t0 = time.perf_counter()
        results_iter = self.search_client.search(**search_kwargs)
        results = list(results_iter)
        t1 = time.perf_counter()

        if DEBUG:
            print(
                f"Fuzzy Azure Search (filter={filter_expr}) took {t1 - t0:.3f} seconds, "
                f"candidates={len(results)}"
            )

        return results

    # ------------------------------------------------------------------
    # Light scoring for candidates
    # ------------------------------------------------------------------
    def _rank_with_light_scoring(
        self,
        query_dob: str,
        query_mincode: str,
        query_postal: str,
        query_sex: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []

        for cand in candidates:
            base_score = cand.get("@search.score", 0.0)

            dob_sim = self._dob_similarity(query_dob, cand.get("dob"))
            mincode_sim = self._mincode_similarity(
                query_mincode, cand.get("mincode", "") or ""
            )
            postal_sim = self._postal_similarity(
                query_postal, cand.get("postalCode", "") or ""
            )
            sex_sim = self._sex_similarity(
                query_sex, (cand.get("sexCode") or "").upper()
            )

            dob_match = dob_sim == 1.0
            mincode_match = mincode_sim == 1.0
            postal_match = postal_sim >= 0.7
            sex_match = sex_sim == 1.0

            hard_boost = 0.0
            if dob_match and mincode_match:
                hard_boost = 1.0
            elif dob_match or mincode_match:
                hard_boost = 0.5
            elif postal_match:
                hard_boost = 0.2
            elif sex_match:
                hard_boost = 0.1

            final_score = base_score + hard_boost

            cand["base_search_score"] = base_score
            cand["hard_boost"] = hard_boost
            cand["final_score"] = final_score

            ranked.append(cand)

        ranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # Cascading-filter fuzzy search
    # ------------------------------------------------------------------
    def soft_fuzzy_search(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuzzy search with cascading hard filters + name embedding:
        1. Build name embedding from query.
        2. Try increasingly loose filters, e.g.:
           - dob & sex & mincode
           - dob & sex
           - mincode & sex
           - postal FSA & sex
           - sex only
           - no filter
        3. For the first filter that returns candidates, rank them by:
           final_score = @search.score (vector) + hard_boost(DOB/mincode/postal/sex).
        4. Return top 20.
        """

        # 1. Build embedding from name
        query_embedding = self.generate_embedding(query_data)
        if not query_embedding:
            return {
                "results": [],
                "count": 0,
                "methodology": {
                    "reason": "No name embedding available for fuzzy search",
                },
            }

        # 2. Normalize query fields
        q_dob = self._normalize_query_dob(
            query_data.get("dob")
            or query_data.get("birthDate")
            or ""
        )
        q_mincode = (query_data.get("mincode") or "").strip()
        q_postal = (
            query_data.get("postalCode")
            or query_data.get("postal")
            or ""
        ).strip()
        q_sex = (
            query_data.get("sexCode")
            or query_data.get("sex")
            or ""
        ).upper()

        # Build filter list (from strict to loose)
        filters: List[Optional[str]] = []
        filter_labels: List[str] = []

        # dob & sex & mincode
        if q_dob and q_sex and q_mincode:
            expr = (
                f"dob eq {self._dob_literal(q_dob)} "
                f"and sexCode eq '{self._escape_filter_str(q_sex)}' "
                f"and mincode eq '{self._escape_filter_str(q_mincode)}'"
            )
            filters.append(expr)
            filter_labels.append("dob+sex+mincode")

        # dob & sex
        if q_dob and q_sex:
            expr = (
                f"dob eq {self._dob_literal(q_dob)} "
                f"and sexCode eq '{self._escape_filter_str(q_sex)}'"
            )
            filters.append(expr)
            filter_labels.append("dob+sex")

        # mincode & sex
        if q_mincode and q_sex:
            expr = (
                f"mincode eq '{self._escape_filter_str(q_mincode)}' "
                f"and sexCode eq '{self._escape_filter_str(q_sex)}'"
            )
            filters.append(expr)
            filter_labels.append("mincode+sex")

        # postal FSA & sex
        if q_postal and q_sex:
            fsa = q_postal.replace(" ", "").upper()[:3]
            if fsa:
                expr = (
                    f"startswith(postalCode, '{self._escape_filter_str(fsa)}') "
                    f"and sexCode eq '{self._escape_filter_str(q_sex)}'"
                )
                filters.append(expr)
                filter_labels.append("postalFSA+sex")

        # sex only
        if q_sex:
            expr = f"sexCode eq '{self._escape_filter_str(q_sex)}'"
            filters.append(expr)
            filter_labels.append("sex")

        # final fallback: no filter
        filters.append(None)
        filter_labels.append("none")

        chosen_filter_expr: Optional[str] = None
        chosen_filter_label: str = ""
        raw_candidates: List[Dict[str, Any]] = []
        top_k_vector = 300  # bigger pool to avoid cutting off correct record

        # 3. Cascade through filters until we get some candidates
        for expr, label in zip(filters, filter_labels):
            candidates = self._vector_search_once(
                embedding=query_embedding,
                filter_expr=expr,
                top_k=top_k_vector,
            )
            if candidates:
                chosen_filter_expr = expr
                chosen_filter_label = label
                raw_candidates = candidates
                break

        if not raw_candidates:
            return {
                "results": [],
                "count": 0,
                "methodology": {
                    "reason": "No candidates found across all cascade filters",
                    "filters_tried": filter_labels,
                },
            }

        # 4. Rank with light scoring and keep top 20
        ranked_candidates = self._rank_with_light_scoring(
            query_dob=q_dob,
            query_mincode=q_mincode,
            query_postal=q_postal,
            query_sex=q_sex,
            candidates=raw_candidates,
        )

        top_candidates = ranked_candidates[:20]

        methodology = {
            "search_method": "vector_only_with_cascading_filters",
            "embedding_generated": query_embedding is not None,
            "filters_tried": filter_labels,
            "chosen_filter_label": chosen_filter_label,
            "chosen_filter_expr": chosen_filter_expr,
            "vector_top_k": top_k_vector,
            "candidates_before_ranking": len(raw_candidates),
            "candidates_after_ranking": len(top_candidates),
        }

        return {
            "results": top_candidates,
            "count": len(top_candidates),
            "methodology": methodology,
        }
