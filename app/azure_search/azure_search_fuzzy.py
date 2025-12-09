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
        """If you ever need exact DOB equality in filters."""
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

        Scoring:
          1.0  → exact date match
          0.7  → same year+month
          0.4  → same year
          0.0  → otherwise
        """
        if not query_dob or not candidate_dob:
            return 0.0

        if hasattr(candidate_dob, "strftime"):
            cand_full = candidate_dob.strftime("%Y-%m-%d")
        else:
            cand_full = str(candidate_dob)[:10]

        if query_dob == cand_full:
            return 1.0

        q_year = query_dob[:4]
        q_year_month = query_dob[:7]
        c_year = cand_full[:4]
        c_year_month = cand_full[:7]

        if q_year_month == c_year_month:
            return 0.7
        if q_year == c_year:
            return 0.4
        return 0.0

    # ------------------------------------------------------------------
    # Helpers: prefix → range filters (because startswith is not allowed)
    # ------------------------------------------------------------------
    def _build_dob_month_range_filter(self, q_dob: str) -> Optional[str]:
        """
        Build a DOB month range:
          dob in [YYYY-MM-01, YYYY-MM-32)
        """
        if not q_dob:
            return None
        dob_prefix = q_dob[:7]  # 'YYYY-MM'
        start = f"{dob_prefix}-01"
        end = f"{dob_prefix}-32"  # 32 is > any valid day, still < next month
        start_esc = self._escape_filter_str(start)
        end_esc = self._escape_filter_str(end)
        return f"dob ge '{start_esc}' and dob lt '{end_esc}'"

    def _build_mincode_prefix_range_filter(self, q_mincode: str) -> Optional[str]:
        """
        Build a MINCODE prefix range:
          mincode in [prefix, prefix_high)
        where prefix_high = prefix+1 in numeric space (with same length).
        """
        q_mincode = (q_mincode or "").strip()
        if not q_mincode:
            return None

        # Use first 4 digits if possible, otherwise first 3
        prefix_len = 4 if len(q_mincode) >= 4 else len(q_mincode)
        if prefix_len < 3:
            return None  # too broad

        prefix = q_mincode[:prefix_len]
        if not prefix.isdigit():
            # Fallback: no fancy range if not all digits
            prefix_esc = self._escape_filter_str(prefix)
            return f"mincode ge '{prefix_esc}'"

        prefix_int = int(prefix)
        next_int = prefix_int + 1
        prefix_high = str(next_int).zfill(prefix_len)

        prefix_esc = self._escape_filter_str(prefix)
        prefix_high_esc = self._escape_filter_str(prefix_high)
        return f"mincode ge '{prefix_esc}' and mincode lt '{prefix_high_esc}'"

    def _build_postal_fsa_range_filter(self, q_postal: str) -> Optional[str]:
        """
        Build a POSTAL FSA range based on first 3 chars:
          postalCode in [FSA, FSA_next)
        where FSA_next is the next last-letter (e.g. V8W → V8X).
        If last char is 'Z', we just do ge FSA.
        """
        if not q_postal:
            return None

        fsa = q_postal.replace(" ", "").upper()[:3]
        if not fsa:
            return None

        first_two = fsa[:2]
        last_char = fsa[2]

        if "A" <= last_char < "Z":
            next_last = chr(ord(last_char) + 1)
            fsa_high = first_two + next_last
            fsa_esc = self._escape_filter_str(fsa)
            fsa_high_esc = self._escape_filter_str(fsa_high)
            return f"postalCode ge '{fsa_esc}' and postalCode lt '{fsa_high_esc}'"
        else:
            # If 'Z' or something unexpected, fallback to ge only
            fsa_esc = self._escape_filter_str(fsa)
            return f"postalCode ge '{fsa_esc}'"

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
                exhaustive=False,  # could be True for exact ranking, slower
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
    # Light scoring for final union of candidates
    # ------------------------------------------------------------------
    def _rank_with_light_scoring(
        self,
        query_dob: str,
        query_mincode: str,
        query_postal: str,
        query_sex: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Final ranking:
          final_score =
              base_score (name embedding)
            + 0.5 * dob_sim   (DOB most important)
            + 0.3 * mincode_sim
            + 0.2 * postal_sim
            + 0.1 * sex_sim
        """
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

            final_score = (
                base_score
                + 0.5 * dob_sim
                + 0.3 * mincode_sim
                + 0.2 * postal_sim
                + 0.1 * sex_sim
            )

            cand["base_search_score"] = base_score
            cand["dob_sim"] = dob_sim
            cand["mincode_sim"] = mincode_sim
            cand["postal_sim"] = postal_sim
            cand["sex_sim"] = sex_sim
            cand["final_score"] = final_score

            ranked.append(cand)

        ranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # New fuzzy search: sex filter + coarse DOB/MINCODE/POSTAL filters (via ranges)
    # ------------------------------------------------------------------
    def soft_fuzzy_search(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuzzy search with:
        - Name embedding as main signal.
        - Optional global sex filter.
        - Three separate filtered vector searches:
            * DOB month range filter + optional sex
            * MINCODE prefix numeric range filter + optional sex
            * POSTAL FSA range filter + optional sex
        - Union of candidates from those searches (deduped).
        - If all three are empty, fallback to:
            * sex-only filter, then
            * no filter.
        - Final ranking = name embedding score + DOB/MINCODE/POSTAL/sex boosts.

        Goal:
          last name + first name + ANY “almost correct” extra field
          (dob, mincode, postal) should surface the right PEN,
          even if the last character of each field is wrong.
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

        # Sex filter (optional)
        sex_filter_expr = None
        if q_sex:
            sex_filter_expr = f"sexCode eq '{self._escape_filter_str(q_sex)}'"

        filters_run: List[str] = []
        candidates_all: List[Dict[str, Any]] = []
        top_k_vector = 300  # large enough so name+field won't be cut off in most cases

        # 3. DOB month range filter + optional sex
        dob_range_expr = self._build_dob_month_range_filter(q_dob)
        if dob_range_expr:
            dob_expr = dob_range_expr
            if sex_filter_expr:
                dob_expr = f"{dob_expr} and {sex_filter_expr}"

            filters_run.append(f"DOB_RANGE filter: {dob_expr}")
            dob_candidates = self._vector_search_once(
                embedding=query_embedding,
                filter_expr=dob_expr,
                top_k=top_k_vector,
            )
            candidates_all.extend(dob_candidates)

        # 4. MINCODE prefix numeric range filter + optional sex
        min_range_expr = self._build_mincode_prefix_range_filter(q_mincode)
        if min_range_expr:
            min_expr = min_range_expr
            if sex_filter_expr:
                min_expr = f"{min_expr} and {sex_filter_expr}"

            filters_run.append(f"MINCODE_RANGE filter: {min_expr}")
            min_candidates = self._vector_search_once(
                embedding=query_embedding,
                filter_expr=min_expr,
                top_k=top_k_vector,
            )
            candidates_all.extend(min_candidates)

        # 5. POSTAL FSA range filter + optional sex
        post_range_expr = self._build_postal_fsa_range_filter(q_postal)
        if post_range_expr:
            post_expr = post_range_expr
            if sex_filter_expr:
                post_expr = f"{post_expr} and {sex_filter_expr}"

            filters_run.append(f"POSTAL_FSA_RANGE filter: {post_expr}")
            post_candidates = self._vector_search_once(
                embedding=query_embedding,
                filter_expr=post_expr,
                top_k=top_k_vector,
            )
            candidates_all.extend(post_candidates)

        # 6. If all three filtered lists are empty, use fallback(s)
        if not candidates_all:
            # sex-only fallback
            if sex_filter_expr:
                filters_run.append(f"SEX-ONLY filter: {sex_filter_expr}")
                sex_candidates = self._vector_search_once(
                    embedding=query_embedding,
                    filter_expr=sex_filter_expr,
                    top_k=top_k_vector,
                )
                candidates_all.extend(sex_candidates)

            # final fallback: no filter
            if not candidates_all:
                filters_run.append("NO filter")
                nofilter_candidates = self._vector_search_once(
                    embedding=query_embedding,
                    filter_expr=None,
                    top_k=top_k_vector,
                )
                candidates_all.extend(nofilter_candidates)

        if not candidates_all:
            # truly nothing found
            return {
                "results": [],
                "count": 0,
                "methodology": {
                    "reason": "No candidates found from all filters and fallbacks",
                    "filters_run": filters_run,
                },
            }

        # 7. Deduplicate by student_id (preferred) or pen
        combined: Dict[str, Dict[str, Any]] = {}
        for cand in candidates_all:
            sid = cand.get("student_id") or cand.get("pen")
            if not sid:
                # fallback key if absolutely needed
                sid = cand.get("@search.documentKey") or str(id(cand))

            base_score = cand.get("@search.score", 0.0)
            existing = combined.get(sid)

            if not existing:
                cand["@search.score"] = base_score
                combined[sid] = cand
            else:
                # keep the candidate with the highest base (embedding) score
                if base_score > existing.get("@search.score", 0.0):
                    cand["@search.score"] = base_score
                    combined[sid] = cand

        unique_candidates = list(combined.values())

        # 8. Final ranking (name embedding + DOB/MINCODE/POSTAL/sex boosts)
        ranked_candidates = self._rank_with_light_scoring(
            query_dob=q_dob,
            query_mincode=q_mincode,
            query_postal=q_postal,
            query_sex=q_sex,
            candidates=unique_candidates,
        )

        top_candidates = ranked_candidates[:20]

        methodology = {
            "search_method": "vector_only_with_sex_and_coarse_field_range_filters",
            "embedding_generated": query_embedding is not None,
            "filters_run": filters_run,
            "vector_top_k_per_filter": top_k_vector,
            "candidates_before_dedup": len(candidates_all),
            "candidates_after_dedup": len(unique_candidates),
            "candidates_returned": len(top_candidates),
        }

        return {
            "results": top_candidates,
            "count": len(top_candidates),
            "methodology": methodology,
        }
