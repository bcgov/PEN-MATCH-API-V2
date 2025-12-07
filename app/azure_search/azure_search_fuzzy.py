from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
import time
from typing import Dict, List, Any, Optional

from config.settings import settings


class FuzzySearchService:
    """Service for fuzzy/vector-based student search operations."""
    
    def __init__(self, search_endpoint: str, index_name: str, credential):
        self.search_endpoint = search_endpoint
        self.index_name = index_name
        self.credential = credential
        
        # OpenAI embedding client
        self.openai_client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version="2023-05-15",
            azure_endpoint=settings.openai_api_base_embedding
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
        """Generate embedding from full name using text-embedding-3-large."""
        first = (query_data.get("legalFirstName") or "").strip()
        last = (query_data.get("legalLastName") or "").strip()
        middle = (query_data.get("legalMiddleNames") or "").strip()

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
            print(f"Generating embedding for: {text}")
            t0 = time.perf_counter()
            resp = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            t1 = time.perf_counter()
            print(f"Embedding generation took {t1 - t0:.3f} seconds")
            return resp.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for '{text}': {e}")
            return None

    # ------------------------------------------------------------------
    # Similarity helpers (NO difflib)
    # ------------------------------------------------------------------
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

        # same first 2 chars (very rough, but better than 0)
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

        # prefix-based soft match
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

    def _dob_similarity(self, query_dob: str, candidate_dob: str) -> float:
        if not query_dob or not candidate_dob:
            return 0.0
        # exact date string match (ISO yyyy-mm-dd)
        return 1.0 if query_dob == candidate_dob else 0.0

    # ------------------------------------------------------------------
    # Soft fuzzy search: vector-only on nameEmbedding (HNSW)
    # Vector search keeps top 200 → soft scoring keeps top 20
    # ------------------------------------------------------------------
    def soft_fuzzy_search(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuzzy search with:
        - vector-only search on nameEmbedding (HNSW, keep top 200)
        - NO text search on content or name fields
        - metadata-based soft scoring (DOB, mincode, postal, sex)
        - after scoring, keep final top 20

        This runs ONLY if hard filter returned 0 results.
        """
        # 1. Build embedding from name (only entry for fuzzy)
        query_embedding = self.generate_embedding(query_data)

        if not query_embedding:
            return {
                "results": [],
                "count": 0,
                "methodology": {
                    "reason": "No name embedding available for fuzzy search",
                },
            }

        # 2. Build vector query (HNSW, approximate, top 200)
        vector_queries: List[VectorizedQuery] = [
            VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=200,   # 🔹 return top 200 neighbors from vector index
                fields="nameEmbedding",
                exhaustive=False           # use HNSW approximate search
            )
        ]

        try:
            # 3. Vector-only search (no text search_text, no search_fields)
            t0 = time.perf_counter()
            search_method = "vector_only"
            results = self.search_client.search(
                search_text="*",               # required param, but not used for ranking
                vector_queries=vector_queries,
                top=200,                       # 🔹 we pull top 200 docs back
                include_total_count=False,     # faster, no total count
            )
            candidates_list = list(results)
            t1 = time.perf_counter()
            print(
                f"Fuzzy Azure Search ({search_method}) took {t1 - t0:.3f} seconds "
                # f"returned {len(candidates_list)} rows"
            )

            # 4. Soft filtering: ONLY by sex (if provided)
            filtered_candidates = []
            query_sex = (query_data.get("sexCode") or "").upper()

            for cand in candidates_list:
                cand_sex = (cand.get("sexCode") or "").upper()
                # if user provided sex, candidate must match
                if query_sex and cand_sex and cand_sex != query_sex:
                    continue
                filtered_candidates.append(cand)

            # 5. Compute soft scores and final scores
            scored_candidates = []
            for cand in filtered_candidates:
                base_score = cand.get("@search.score", 0.0)

                dob_sim = self._dob_similarity(
                    query_data.get("dob", ""),
                    cand.get("dob", ""),
                )
                postal_sim = self._postal_similarity(
                    query_data.get("postalCode", ""),
                    cand.get("postalCode", ""),
                )
                mincode_sim = self._mincode_similarity(
                    query_data.get("mincode", ""),
                    cand.get("mincode", ""),
                )
                sex_sim = self._sex_similarity(
                    query_data.get("sexCode", ""),
                    cand.get("sexCode", ""),
                )

                # Default weights (importance order)
                field_weights = {
                    "dob": 0.4,
                    "mincode": 0.3,
                    "postal": 0.2,
                    "sex": 0.1,
                }

                # Only use weights for fields actually provided in query
                active_weights = {}
                if query_data.get("dob"):
                    active_weights["dob"] = field_weights["dob"]
                if query_data.get("mincode"):
                    active_weights["mincode"] = field_weights["mincode"]
                if query_data.get("postalCode"):
                    active_weights["postal"] = field_weights["postal"]
                if query_data.get("sexCode"):
                    active_weights["sex"] = field_weights["sex"]

                total_weight = sum(active_weights.values())
                soft_score = 0.0
                if total_weight > 0:
                    if "dob" in active_weights:
                        soft_score += dob_sim * (active_weights["dob"] / total_weight)
                    if "mincode" in active_weights:
                        soft_score += mincode_sim * (active_weights["mincode"] / total_weight)
                    if "postal" in active_weights:
                        soft_score += postal_sim * (active_weights["postal"] / total_weight)
                    if "sex" in active_weights:
                        soft_score += sex_sim * (active_weights["sex"] / total_weight)

                # If no extra fields provided, we rely purely on base score
                if total_weight == 0:
                    final_score = base_score
                else:
                    final_score = 0.3 * base_score + soft_score

                cand["base_search_score"] = base_score
                cand["soft_score"] = soft_score
                cand["final_score"] = final_score
                cand["search_method"] = search_method

                scored_candidates.append(cand)

            # 6. Sort by final score and keep top 20
            scored_candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
            top_candidates = scored_candidates[:20]

            # Debug print top 5
            # print(f"\n=== DEBUG: TOP 5 FUZZY SEARCH CANDIDATES ({search_method.upper()}) ===")
            # for i, cand in enumerate(top_candidates[:5], 1):
            #     middle = cand.get("legalMiddleNames") or ""
            #     print(
            #         f"{i}. {cand.get('legalFirstName', '')} {middle} {cand.get('legalLastName', '')}"
            #     )
            #     print(f"   Sex: {cand.get('sexCode', '')}, DOB: {cand.get('dob', '')}")
            #     print(
            #         f"   Postal: {cand.get('postalCode', '')}, Mincode: {cand.get('mincode', '')}"
            #     )
            #     print(
            #         f"   Grade: {cand.get('gradeCode', '')}, Local ID: {cand.get('localID', '')}"
            #     )
            #     print(
            #         f"   Base Score: {cand.get('base_search_score', 0):.4f}, "
            #         f"Soft Score: {cand.get('soft_score', 0):.4f}, "
            #         f"Final Score: {cand.get('final_score', 0):.4f}"
            #     )
            #     print()

            methodology = {
                "step1": "Azure Search with vector-only HNSW on nameEmbedding (top 200)",
                "step2": "Sex filter only (if provided)",
                "step3": "Soft scoring on DOB, mincode, postal, sex with dynamic weights",
                "step4": "Final ranking by 0.3 * base_score + soft_score (keep top 20)",
                "search_method": search_method,
                "embedding_generated": query_embedding is not None,
                "candidates_initial": len(candidates_list),
                "candidates_after_sex_filter": len(filtered_candidates),
                "candidates_after_scoring": len(top_candidates),
            }

            return {
                "results": top_candidates,
                "count": len(top_candidates),
                "methodology": methodology,
            }

        except Exception as e:
            print(f"Error in soft fuzzy search: {str(e)}")
            return {
                "results": [],
                "count": 0,
                "methodology": {"error": str(e)},
            }