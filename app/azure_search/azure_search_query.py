from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
import difflib
import time
from typing import Dict, List, Any, Optional

from config.settings import settings


class StudentSearchService:
    def __init__(self):
        self.search_endpoint = "https://pen-match-api-v2-search.search.windows.net"
        self.index_name = "student-index"

        # Azure identity
        self.credential = DefaultAzureCredential()

        # OpenAI embedding client
        self.openai_client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version="2023-05-15",
            azure_endpoint=settings.openai_api_base_embedding_3,
        )

        try:
            self.search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.index_name,
                credential=self.credential,
            )
            print("Azure Search client initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Azure Search client: {str(e)}")
            raise

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
                model="text-embedding-3-large",
                input=text,
                dimensions=3072,
            )
            t1 = time.perf_counter()
            print(f"Embedding generation took {t1 - t0:.3f} seconds")
            return resp.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for '{text}': {e}")
            return None

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------
    def _postal_similarity(self, query_postal: str, candidate_postal: str) -> float:
        if not query_postal or not candidate_postal:
            return 0.0

        q = query_postal.replace(" ", "").upper()
        c = candidate_postal.replace(" ", "").upper()

        if q == c:
            return 1.0

        # same FSA (first 3 chars)
        if len(q) >= 3 and len(c) >= 3 and q[:3] == c[:3]:
            return 0.7

        sim = difflib.SequenceMatcher(None, q, c).ratio()
        return sim if sim > 0.5 else 0.0

    def _mincode_similarity(self, query_mincode: str, candidate_mincode: str) -> float:
        if not query_mincode or not candidate_mincode:
            return 0.0

        q = str(query_mincode).strip()
        c = str(candidate_mincode).strip()

        if q == c:
            return 1.0

        sim = difflib.SequenceMatcher(None, q, c).ratio()
        return sim if sim > 0.8 else 0.0

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
    # Hard filter search (exact matching with OData filter)
    # ------------------------------------------------------------------
    def _hard_filter_search(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform strict hard filtering.
        If user gives multiple fields, we do AND between them.
        This is where 'exact match' lives. If this returns >0, we never go to fuzzy.
        (NO embedding is used in this path.)
        """
        filters = []

        if query_data.get("pen"):
            filters.append(f"pen eq '{query_data['pen']}'")

        if query_data.get("legalFirstName"):
            filters.append(f"legalFirstName eq '{query_data['legalFirstName']}'")

        if query_data.get("legalLastName"):
            filters.append(f"legalLastName eq '{query_data['legalLastName']}'")

        if query_data.get("legalMiddleNames"):
            filters.append(f"legalMiddleNames eq '{query_data['legalMiddleNames']}'")

        if query_data.get("dob"):
            filters.append(f"dob eq '{query_data['dob']}'")

        if query_data.get("postalCode"):
            filters.append(f"postalCode eq '{query_data['postalCode']}'")

        if query_data.get("mincode"):
            filters.append(f"mincode eq '{query_data['mincode']}'")

        if query_data.get("gradeCode"):
            filters.append(f"gradeCode eq '{query_data['gradeCode']}'")

        if query_data.get("localID"):
            filters.append(f"localID eq '{query_data['localID']}'")

        filter_expression = " and ".join(filters) if filters else None

        try:
            t0 = time.perf_counter()
            results = self.search_client.search(
                search_text="*",
                filter=filter_expression,
                top=101,  # fetch 101 to detect >100
                include_total_count=True,
            )
            results_list = list(results)
            t1 = time.perf_counter()

            count = len(results_list)
            print(
                f"Exact search (hard filter) took {t1 - t0:.3f} seconds, "
                f"returned {count} rows"
            )

            return {
                "results": results_list[:100],
                "count": count,
            }
        except Exception as e:
            print(f"Error in hard filter search: {str(e)}")
            return {"results": [], "count": 0}

    # ------------------------------------------------------------------
    # Soft fuzzy search: vector-only on nameEmbedding (HNSW), top 20
    # ------------------------------------------------------------------
    def _soft_fuzzy_search(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuzzy search with:
        - vector-only search on nameEmbedding (HNSW, top 20)
        - NO text search on content or name fields
        - metadata-based soft scoring (DOB, mincode, postal, sex)

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

        # 2. Build vector query (HNSW, approximate, top 20)
        vector_queries: List[VectorizedQuery] = [
            VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=20,   # only top 20 neighbors from vector index
                fields="nameEmbedding",
                exhaustive=False          # use HNSW approximate search
            )
        ]

        try:
            # 3. Vector-only search (no text search_text, no search_fields)
            t0 = time.perf_counter()
            search_method = "vector_only"
            results = self.search_client.search(
                search_text="*",              # required param, but not used for ranking
                vector_queries=vector_queries,
                top=20,                       # we only need 20 docs back
                include_total_count=False,    # faster, no total count
            )
            candidates_list = list(results)
            t1 = time.perf_counter()
            print(
                f"Fuzzy Azure Search ({search_method}) took {t1 - t0:.3f} seconds, "
                f"returned {len(candidates_list)} rows"
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
            print(f"\n=== DEBUG: TOP 5 FUZZY SEARCH CANDIDATES ({search_method.upper()}) ===")
            for i, cand in enumerate(top_candidates[:5], 1):
                middle = cand.get("legalMiddleNames") or ""
                print(
                    f"{i}. {cand.get('legalFirstName', '')} {middle} {cand.get('legalLastName', '')}"
                )
                print(f"   Sex: {cand.get('sexCode', '')}, DOB: {cand.get('dob', '')}")
                print(
                    f"   Postal: {cand.get('postalCode', '')}, Mincode: {cand.get('mincode', '')}"
                )
                print(
                    f"   Grade: {cand.get('gradeCode', '')}, Local ID: {cand.get('localID', '')}"
                )
                print(
                    f"   Base Score: {cand.get('base_search_score', 0):.4f}, "
                    f"Soft Score: {cand.get('soft_score', 0):.4f}, "
                    f"Final Score: {cand.get('final_score', 0):.4f}"
                )
                print()

            methodology = {
                "step1": "Azure Search with vector-only HNSW on nameEmbedding (top 20)",
                "step2": "Sex filter only (if provided)",
                "step3": "Soft scoring on DOB, mincode, postal, sex with dynamic weights",
                "step4": "Final ranking by 0.3 * base_score + soft_score",
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

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def search_students(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main search pipeline:
        1. Hard filter (exact). If 1–100 hits → return (NO embedding).
           If >100 → error.
        2. If 0 → soft fuzzy search with vector-only HNSW + soft score.
        """
        # STEP 1: hard filter (no embedding here)
        hard = self._hard_filter_search(query_data)

        if hard["count"] > 100:
            return {
                "status": "error",
                "message": "Over 100 candidates found, need more specific information",
                "count": hard["count"],
                "search_type": "exact_match",
                "results": [],
            }

        if hard["count"] > 0:
            return {
                "status": "success",
                "results": hard["results"],
                "count": hard["count"],
                "search_type": "exact_match",
            }

        # STEP 2: soft fuzzy search
        fuzzy = self._soft_fuzzy_search(query_data)
        return {
            "status": "success",
            "results": fuzzy["results"],
            "count": fuzzy["count"],
            "search_type": "fuzzy_match",
            "methodology": fuzzy.get("methodology", {}),
        }


# ----------------------------------------------------------------------
# Helper wrapper
# ----------------------------------------------------------------------
def search_student_by_query(query_data: Dict[str, Any]) -> Dict[str, Any]:
    service = StudentSearchService()
    return service.search_students(query_data)


# ----------------------------------------------------------------------
# Pretty printing of results (for debugging and tests)
# ----------------------------------------------------------------------
def print_search_results(result: Dict[str, Any], max_display: int = 5):
    print("\n=== SEARCH RESULTS ===")
    print(f"Status: {result.get('status')}")
    print(f"Search Type: {result.get('search_type', '').upper()}")
    print(f"Total Count: {result.get('count', 0)}")

    if result.get("status") == "error":
        print(f"Error Message: {result.get('message')}")
        return

    count = result.get("count", 0)
    if count == 0:
        print("No candidates found")
        return

    print(f"\nShowing top {min(max_display, count)} candidates:")
    print("-" * 80)

    for i, cand in enumerate(result["results"][:max_display], 1):
        middle = cand.get("legalMiddleNames", "") or ""
        print(f"\n{i}. Student ID: {cand.get('student_id', 'N/A')}")
        print(f"   PEN: {cand.get('pen', 'N/A')}")
        print(
            f"   Name: {cand.get('legalFirstName', '')} {middle} {cand.get('legalLastName', '')}".strip()
        )
        print(f"   DOB: {cand.get('dob', 'N/A')}")
        print(f"   Sex: {cand.get('sexCode', 'N/A')}")
        print(f"   Postal: {cand.get('postalCode', 'N/A')}")
        print(f"   Mincode: {cand.get('mincode', 'N/A')}")
        print(f"   Grade: {cand.get('gradeCode', 'N/A')}")
        print(f"   LocalID: {cand.get('localID', 'N/A')}")

        if result.get("search_type") == "fuzzy_match":
            print(f"   Base Score: {cand.get('base_search_score', 'N/A')}")
            print(f"   Soft Score: {cand.get('soft_score', 'N/A')}")
            print(f"   Final Score: {cand.get('final_score', 'N/A')}")
            print(f"   Search Method: {cand.get('search_method', 'N/A')}")


# ----------------------------------------------------------------------
# Test Suite
# ----------------------------------------------------------------------
def run_test_suite():
    print("=" * 60)
    print("AZURE SEARCH TEST SUITE (WITH VECTOR EMBEDDING)")
    print("=" * 60)

    # EXACT MATCH TESTS
    print("\nEXACT MATCH TESTS")
    print("=" * 40)

    exact_test_cases = [
        {
            "name": "Exact Match - PEN Only",
            "query": {
                "pen": "124809765",
            },
        },
        {
            # still goes through HARD filter (all fields exact) and should be exact_match
            "name": "Exact Search - Full Info",
            "query": {
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE",
                "legalMiddleNames": "RICHARD",
                "dob": "2001-02-10",
                "sexCode": "M",
                "postalCode": "V3N1H4",
                "mincode": "05757079",
            },
        },
        {
            "name": "Exact Match - Full Name",
            "query": {
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE",
                "legalMiddleNames": "RICHARD",
            },
        },
        {
            "name": "Exact Match - Name + DOB",
            "query": {
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE",
                "dob": "2001-02-10",
            },
        },
    ]

    for i, tc in enumerate(exact_test_cases, 1):
        print(f"\nTest {i}: {tc['name']}")
        print(f"Query: {tc['query']}")
        result = search_student_by_query(tc["query"])
        # print_search_results(result, max_display=5)

    # FUZZY MATCH TESTS
    print("\n\nFUZZY MATCH TESTS (VECTOR-ONLY HNSW, TOP 20)")
    print("=" * 40)

    fuzzy_test_cases = [
        {
            "name": "Fuzzy Search - Typo Name",
            "query": {
                "legalFirstName": "MICHEAL",  # Typo
                "legalLastName": "LEE",
                "legalMiddleNames": "RICHARD",
            },
        },
        {
            "name": "Fuzzy Search - Typo DOB",
            "query": {
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE",
                "legalMiddleNames": "RICHARD",
                "dob": "2010-02-10",  # wrong year
            },
        },
        {
            "name": "Fuzzy Search - Typo Postal code",
            "query": {
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE",
                "postalCode": "V3N4H1",  # typo
                "mincode": "05757079",
            },
        },
        {
            "name": "Fuzzy Search - Typo mincode",
            "query": {
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE",
                "postalCode": "V3N4H1",  # typo in postal as well
                "mincode": "0575079",    # typo in mincode
            },
        },
        {
            "name": "Fuzzy Search - Typo Full Info",
            "query": {
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE",
                "legalMiddleNames": "RICH",  # shortened
                "dob": "2001-02-10",
                "sexCode": "M",
                "postalCode": "V3N1H5",
                "mincode": "5757079",
            },
        },
    ]

    for i, tc in enumerate(fuzzy_test_cases, 1):
        print(f"\nFuzzy Test {i}: {tc['name']}")
        print(f"Query: {tc['query']}")
        result = search_student_by_query(tc["query"])
        # print_search_results(result, max_display=5)


if __name__ == "__main__":
    run_test_suite()
