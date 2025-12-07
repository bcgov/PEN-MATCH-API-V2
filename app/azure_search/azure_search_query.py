from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
import time
from typing import Dict, List, Any

from config.settings import settings
from .azure_search_fuzzy import FuzzySearchService


class StudentSearchService:
    def __init__(self):
        self.search_endpoint = "https://pen-match-api-v2-search.search.windows.net"
        self.index_name = "student-index"

        # Azure identity
        self.credential = DefaultAzureCredential()

        try:
            self.search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.index_name,
                credential=self.credential,
            )
            print("Azure Search client initialized successfully")
            
            # Initialize fuzzy search service
            self.fuzzy_service = FuzzySearchService(
                self.search_endpoint,
                self.index_name,
                self.credential
            )
        except Exception as e:
            print(f"Failed to initialize Azure Search client: {str(e)}")
            raise

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

        # STEP 2: soft fuzzy search (delegated to fuzzy service)
        fuzzy = self.fuzzy_service.soft_fuzzy_search(query_data)
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
    # print(f"Total Count: {result.get('count', 0)}")

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
        first = cand.get("legalFirstName", "") or ""
        middle = cand.get("legalMiddleNames", "") or ""
        last = cand.get("legalLastName", "") or ""

        full_name = " ".join(p for p in [first, middle, last] if p)

        print(f"\n{i}. {full_name}")
        print(f"   Sex: {cand.get('sexCode', 'N/A')}, DOB: {cand.get('dob', 'N/A')}")
        print(
            f"   Postal: {cand.get('postalCode', 'N/A')}, "
            f"Mincode: {cand.get('mincode', 'N/A')}"
        )
        print(
            f"   Grade: {cand.get('gradeCode', 'N/A')}, "
            f"Local ID: {cand.get('localID', 'N/A')}"
        )

        # Extra debug info only for fuzzy match
        if result.get("search_type") == "fuzzy_match":
            print(
                f"   Base Score: {cand.get('base_search_score', 'N/A')}, "
                f"Soft Score: {cand.get('soft_score', 'N/A')}, "
                f"Final Score: {cand.get('final_score', 'N/A')}"
            )
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
        print_search_results(result, max_display=5)

    # FUZZY MATCH TESTS
    print("\n\nFUZZY MATCH TESTS (VECTOR-ONLY HNSW, TOP 200 → TOP 20)")
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
        # Uncomment if you want to see detailed fuzzy output
        # print_search_results(result, max_display=5)


if __name__ == "__main__":
    run_test_suite()