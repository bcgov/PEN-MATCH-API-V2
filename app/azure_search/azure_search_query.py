from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
import time
from typing import Dict, List, Any

from config.settings import settings
from .azure_search_fuzzy import FuzzySearchService

# Only print debug logs when running this module directly (test mode)
DEBUG = __name__ == "__main__"


class StudentSearchService:
    def __init__(self):
        self.search_endpoint = "https://pen-match-api-v2-search.search.windows.net"
        self.index_name = "student-index"

        # Fields we actually use in responses & comparisons
        # (must all exist in the index schema)
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

        # Azure identity
        self.credential = DefaultAzureCredential()

        try:
            self.search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.index_name,
                credential=self.credential,
            )
            if DEBUG:
                print("Azure Search client initialized successfully")

            # Initialize fuzzy search service (reused across calls)
            self.fuzzy_service = FuzzySearchService(
                self.search_endpoint,
                self.index_name,
                self.credential,
            )
        except Exception as e:
            # Init failure is rare but important – keep this print
            print(f"Failed to initialize Azure Search client: {str(e)}")
            raise

    # ------------------------------------------------------------------
    # Helper: search by PEN only (to know if PEN exists)
    # ------------------------------------------------------------------
    def _search_by_pen(self, pen: str) -> Dict[str, Any]:
        """
        Look up student(s) by PEN only.
        We expect at most one record per PEN.
        """
        try:
            t0 = time.perf_counter()
            # Filter-only lookup, minimal fields, no total_count
            results = self.search_client.search(
                search_text="*",  # filter-only pattern
                filter=f"pen eq '{pen}'",
                top=2,  # we only need to know if it exists; 2 detects duplicates
                select=self._select_fields,
            )
            results_list = list(results)
            t1 = time.perf_counter()
            count = len(results_list)
            if DEBUG:
                print(
                    f"[DEBUG] Search by PEN took {t1 - t0:.3f}s, "
                    f"found {count} row(s) for PEN={pen}"
                )
            return {"results": results_list, "count": count}
        except Exception as e:
            if DEBUG:
                print(f"Error in _search_by_pen: {str(e)}")
            return {"results": [], "count": 0}

    # ------------------------------------------------------------------
    # Helper: count how many fields match between query and a record
    # ------------------------------------------------------------------
    @staticmethod
    def _count_matching_fields(query_data: Dict[str, Any], record: Dict[str, Any]) -> (int, int):
        """
        Count how many *provided* fields in query_data match the record.
        Only fields present in query_data are considered.
        Returns (match_count, total_fields_compared).
        """
        compare_fields = [
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

        match_count = 0
        total_fields = 0

        for field in compare_fields:
            qv = query_data.get(field)
            if qv is None or qv == "":
                continue  # not provided → we don't use it for comparison

            rv = record.get(field)
            total_fields += 1

            if rv is not None and str(rv).strip().upper() == str(qv).strip().upper():
                match_count += 1

        if DEBUG:
            print(f"[DEBUG] Field match count: {match_count}/{total_fields}")
        return match_count, total_fields

    # ------------------------------------------------------------------
    # Hard filter search (exact matching with OData filter)
    # ------------------------------------------------------------------
    def _hard_filter_search(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform strict hard filtering.
        If user gives multiple fields, we do AND between them.
        This is where 'exact match' lives. If this returns >0, we don't go to fuzzy.
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

        if DEBUG:
            print(f"[DEBUG] Hard filter expression: {filter_expression}")

        if not filter_expression:
            # No filters → no point calling search (should not happen because
            # StudentQuery always has first/last name, but keep safe)
            return {"results": [], "count": 0}

        try:
            t0 = time.perf_counter()
            # top=41 so we can detect "more than 40" case
            # NOTE: we don't need total_count, we only care about:
            #   - 0
            #   - 1
            #   - 2–40
            #   - >40 (we detect by 41st doc)
            results = self.search_client.search(
                search_text="*",           # filter-only pattern
                filter=filter_expression,
                top=41,
                select=self._select_fields,
            )
            results_list = list(results)
            t1 = time.perf_counter()

            count = len(results_list)
            if DEBUG:
                print(
                    f"[DEBUG] Exact search (hard filter) took {t1 - t0:.3f} seconds, "
                    f"returned {count} rows (top=41)"
                )

            return {
                "results": results_list[:40],  # we only keep at most 40 to return
                "count": count,
            }
        except Exception as e:
            if DEBUG:
                print(f"Error in hard filter search: {str(e)}")
            return {"results": [], "count": 0}

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def search_students(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main search pipeline with PenStatus:

        1. If PEN is provided:
           - Look up by PEN only.
             * Every provided field matches → AA
             * Multiple fields match but some wrong → BM
             * Only one or zero fields match → F1
           - If PEN not found → treat as 'pen not exist' and continue with
             exact + fuzzy using other fields (pen removed).

        2. If no PEN (or PEN not found):
           Exact match (hard filter, no embedding):
             * If 1 candidate     → D1
             * If 2–40 candidates → CM
             * If >40 candidates  → C0 (ask for more info, no list)

           If 0 exact candidates:
             Fuzzy match:
               * For now: top 20 fuzzy candidates, PenStatus = CM if any,
                 or C0 if fuzzy also returns 0.
        """
        if DEBUG:
            print("\n" + "=" * 80)
            print("[DEBUG] Incoming query_data:", query_data)

        pen = query_data.get("pen")
        pen_status = None

        # ------------------------------------------------------------------
        # Case 1: PEN provided → check if PEN exists
        # ------------------------------------------------------------------
        if pen:
            if DEBUG:
                print(f"[DEBUG] PEN supplied in query: {pen}")
            pen_search = self._search_by_pen(pen)
            if pen_search["count"] > 0:
                # PEN exists, compare fields
                record = pen_search["results"][0]  # assume unique PEN
                match_count, total_fields = self._count_matching_fields(query_data, record)

                if total_fields == 0 or match_count == total_fields:
                    # Only PEN given, or all provided fields match
                    pen_status = "AA"
                elif match_count >= 2:
                    pen_status = "BM"
                else:
                    pen_status = "F1"

                if DEBUG:
                    print(
                        f"[DEBUG] PEN lookup pen_status={pen_status}, "
                        f"count={pen_search['count']}"
                    )
                return {
                    "status": "success",
                    "pen_status": pen_status,
                    "search_type": "pen_lookup",
                    "results": pen_search["results"],
                    "count": pen_search["count"],
                }
            else:
                # PEN not found in index → treat as "pen not exist"
                if DEBUG:
                    print(
                        f"[DEBUG] PEN {pen} not found in index, "
                        f"falling back to demographic search."
                    )
                query_no_pen = {k: v for k, v in query_data.items() if k != "pen"}
        else:
            if DEBUG:
                print("[DEBUG] No PEN supplied, using demographic search only.")
            query_no_pen = dict(query_data)

        # ------------------------------------------------------------------
        # Case 2: PEN not provided OR PEN not found → exact match path
        # ------------------------------------------------------------------
        if query_no_pen:
            hard = self._hard_filter_search(query_no_pen)
        else:
            hard = {"results": [], "count": 0}

        count_exact = hard["count"]
        if DEBUG:
            print(f"[DEBUG] Exact match candidate count={count_exact}")

        # > 40 candidates → C0, ask for more info, no list returned
        if count_exact > 40:
            pen_status = "C0"
            if DEBUG:
                print(f"[DEBUG] pen_status={pen_status} (too many exact candidates)")
            return {
                "status": "success",
                "pen_status": pen_status,
                "search_type": "exact_match",
                "message": (
                    "Over 40 candidates found, please provide more "
                    "specific information."
                ),
                "results": [],
                "count": count_exact,
            }

        # 1 candidate → D1
        if count_exact == 1:
            pen_status = "D1"
            if DEBUG:
                print(f"[DEBUG] pen_status={pen_status} (single exact candidate)")
            return {
                "status": "success",
                "pen_status": pen_status,
                "search_type": "exact_match",
                "results": hard["results"],
                "count": count_exact,
            }

        # 2–40 candidates → CM
        if 1 < count_exact <= 40:
            pen_status = "CM"
            if DEBUG:
                print(f"[DEBUG] pen_status={pen_status} (multiple exact candidates)")
            return {
                "status": "success",
                "pen_status": pen_status,
                "search_type": "exact_match",
                "results": hard["results"],
                "count": count_exact,
            }

        # ------------------------------------------------------------------
        # Case 3: No exact candidates → Fuzzy match
        # ------------------------------------------------------------------
        if DEBUG:
            print("[DEBUG] No exact candidates, running fuzzy search...")

        t0_fuzzy = time.perf_counter()
        fuzzy = self.fuzzy_service.soft_fuzzy_search(query_no_pen)
        t1_fuzzy = time.perf_counter()

        fuzzy_count = fuzzy.get("count", 0)
        if DEBUG:
            print(
                f"[DEBUG] Fuzzy match candidate count={fuzzy_count}, "
                f"soft_fuzzy_search total={t1_fuzzy - t0_fuzzy:.3f}s"
            )


        if fuzzy_count == 0:
            # Even fuzzy couldn't find anything
            pen_status = "C0"
            if DEBUG:
                print(f"[DEBUG] pen_status={pen_status} (no fuzzy candidates)")
            return {
                "status": "success",
                "pen_status": pen_status,
                "search_type": "fuzzy_match",
                "results": [],
                "count": 0,
                "methodology": fuzzy.get("methodology", {}),
            }

        # For now: top 20 fuzzy candidates & pen_status = CM
        pen_status = "CM"
        if DEBUG:
            print(f"[DEBUG] pen_status={pen_status} (fuzzy candidates returned)")
        return {
            "status": "success",
            "pen_status": pen_status,
            "search_type": "fuzzy_match",
            "results": fuzzy["results"],   # assume fuzzy already limits to top 20
            "count": fuzzy["count"],
            "methodology": fuzzy.get("methodology", {}),
        }


# ----------------------------------------------------------------------
# Helper wrapper – reuse single service instance to avoid re-auth overhead
# ----------------------------------------------------------------------
student_search_service = StudentSearchService()


def search_student_by_query(query_data: Dict[str, Any]) -> Dict[str, Any]:
    return student_search_service.search_students(query_data)


# ----------------------------------------------------------------------
# Pretty printing of results (for debugging and tests)
# ----------------------------------------------------------------------
def print_search_results(result: Dict[str, Any], max_display: int = 5):
    print("\n=== SEARCH RESULTS ===")
    print(f"Status: {result.get('status')}")
    print(f"Pen Status: {result.get('pen_status', 'N/A')}")
    print(f"Search Type: {result.get('search_type', '').upper()}")

    if result.get("status") == "error":
        print(f"Error Message: {result.get('message')}")
        return

    if "message" in result and result["message"]:
        print(f"Message: {result['message']}")

    count = result.get("count", 0)
    print(f"Total Count: {count}")
    if count == 0:
        print("No candidates found")
        return

    print(f"\nShowing top {min(max_display, count)} candidates:")
    print("-" * 80)

    for i, cand in enumerate(result["results"][:max_display], 1):
        first = cand.get("legalFirstName", "") or ""
        middle = cand.get("legalMiddleNames", "") or ""
        last = cand.get("legalLastName", "") or ""
        pen = cand.get("pen", "N/A")
        student_id = cand.get("student_id", cand.get("studentId", "N/A"))

        full_name = " ".join(p for p in [first, middle, last] if p)

        print(f"\n{i}. {full_name}")
        print(f"   PEN: {pen}, Student ID: {student_id}")
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
        print_search_results(result, max_display=5)


if __name__ == "__main__":
    run_test_suite()
