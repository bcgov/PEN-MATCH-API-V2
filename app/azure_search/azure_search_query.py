from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import os
from typing import Dict, List, Any

class StudentSearchService:
    def __init__(self):
        self.search_endpoint = "https://pen-match-api-v2-search.search.windows.net"
        self.index_name = "student-index"
        
        # Use DefaultAzureCredential for authentication
        self.credential = DefaultAzureCredential()
        
        try:
            self.search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.index_name,
                credential=self.credential
            )
            print("Azure Search client initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Azure Search client: {str(e)}")
            raise
    
    
    def search_students(self, query_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Search for students with hard filtering and fallback to fuzzy search
        """
        # Step 1: Hard filter search
        exact_results = self._hard_filter_search(query_data)
        
        if exact_results["count"] > 100:
            return {
                "status": "error",
                "message": "Over 100 candidates found, need more specific information",
                "count": exact_results["count"],
                "search_type": "exact_match"
            }
        elif exact_results["count"] > 0:
            return {
                "status": "success",
                "results": exact_results["results"],
                "count": exact_results["count"],
                "search_type": "exact_match"
            }
        else:
            # Step 2: Fuzzy search using content
            fuzzy_results = self._fuzzy_content_search(query_data.get("content", ""))
            return {
                "status": "success",
                "results": fuzzy_results["results"],
                "count": fuzzy_results["count"],
                "search_type": "fuzzy_match"
            }
    
    def _hard_filter_search(self, query_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Perform hard filtering by pen→name→dob→postalcode→mincode→grade→localID
        """
        filters = []
        
        # Build filters in priority order
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
            results = self.search_client.search(
                search_text="*",
                filter=filter_expression,
                top=101,  # Get 101 to check if over 100
                include_total_count=True
            )
            
            results_list = list(results)
            return {
                "results": results_list[:100],  # Return max 100
                "count": len(results_list)
            }
        except Exception as e:
            print(f"Error in hard filter search: {str(e)}")
            return {"results": [], "count": 0}
    
    def _fuzzy_content_search(self, content: str) -> Dict[str, Any]:
        """
        Perform fuzzy search using content field with identity scoring
        """
        if not content:
            return {"results": [], "count": 0}
        
        try:
            results = self.search_client.search(
                search_text=content,
                search_fields=["content", "legalFirstName", "legalLastName", "legalMiddleNames"],
                scoring_profile="identityScoring",
                top=100,
                include_total_count=True
            )
            
            results_list = list(results)
            return {
                "results": results_list,
                "count": len(results_list)
            }
        except Exception as e:
            print(f"Error in fuzzy search: {str(e)}")
            return {"results": [], "count": 0}

def search_student_by_query(query_data: Dict[str, str]) -> Dict[str, Any]:
    """
    Main function to search for students
    """
    search_service = StudentSearchService()
    return search_service.search_students(query_data)

def print_search_results(result: Dict[str, Any], max_display: int = 3):
    """
    Print search results in a formatted way
    """
    print(f"\n=== SEARCH RESULTS ===")
    print(f"Status: {result['status']}")
    print(f"Search Type: {result['search_type'].upper()}")
    print(f"Total Count: {result['count']}")
    
    if result['status'] == 'error':
        print(f"Error Message: {result['message']}")
        return
    
    if result['count'] == 0:
        print("No candidates found")
        return
    
    print(f"\nShowing top {min(max_display, result['count'])} candidates:")
    print("-" * 80)
    
    for i, candidate in enumerate(result['results'][:max_display], 1):
        print(f"\n{i}. Student ID: {candidate.get('student_id', 'N/A')}")
        print(f"   PEN: {candidate.get('pen', 'N/A')}")
        print(f"   Name: {candidate.get('legalFirstName', '')} {candidate.get('legalMiddleNames', '') or ''} {candidate.get('legalLastName', '')}".strip())
        print(f"   DOB: {candidate.get('dob', 'N/A')}")
        print(f"   Sex: {candidate.get('sexCode', 'N/A')}")
        print(f"   Postal: {candidate.get('postalCode', 'N/A')}")
        print(f"   Mincode: {candidate.get('mincode', 'N/A')}")
        print(f"   Grade: {candidate.get('gradeCode', 'N/A')}")
        print(f"   LocalID: {candidate.get('localID', 'N/A')}")
        if result['search_type'] == 'fuzzy_match':
            print(f"   Score: {candidate.get('@search.score', 'N/A')}")

# Test cases
def run_test_suite():
    """
    Run test suite with exact and fuzzy search tests
    """
    print("="*60)
    print("AZURE SEARCH TEST SUITE")
    print("="*60)
    
    # EXACT MATCH TESTS
    print("\nEXACT MATCH TESTS")
    print("="*40)
    
    exact_test_cases = [
        {
            "name": "Exact Match - PEN Only",
            "query": {
                "pen": "124809765"
            }
        },
        {
            "name": "Exact Match - First + Last Name",
            "query": {
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE"
            }
        },
        {
            "name": "Exact Match - Full Name",
            "query": {
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE",
                "legalMiddleNames": "RICHARD"
            }
        },
        {
            "name": "Exact Match - Name + DOB",
            "query": {
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE",
                "dob": "2001-02-10"
            }
        },
        {
            "name": "Exact Match - All Fields",
            "query": {
                "pen": "124809765",
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE",
                "legalMiddleNames": "RICHARD",
                "dob": "2001-02-10",
                "sexCode": "M",
                "postalCode": "V3N1H4",
                "mincode": "05757079",
                "gradeCode": "st",
                "localID": "string"
            }
        }
    ]
    
    for i, test_case in enumerate(exact_test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        
        result = search_student_by_query(test_case['query'])
        print_search_results(result, max_display=3)
    
    # FUZZY MATCH TESTS
    print("\n\nFUZZY MATCH TESTS")
    print("="*40)
    
    fuzzy_test_cases = [
        {
            "name": "Fuzzy Search - Similar Name",
            "query": {
                "legalFirstName": "MICHEAL",  # Typo
                "legalLastName": "LEE",
                "content": "MICHEAL LEE, born 2001-02-10, male, postal V3N1H4, mincode 05757079"
            }
        },
        {
            "name": "Fuzzy Search - Partial Info",
            "query": {
                "content": "MICHAEL LEE born 2001-02-10"
            }
        },
        {
            "name": "Fuzzy Search - School Info",
            "query": {
                "content": "MICHAEL LEE mincode 05757079 postal V3N1H4"
            }
        },
        {
            "name": "Fuzzy Search - Common Name",
            "query": {
                "content": "DAVID WANG male"
            }
        },
        {
            "name": "Fuzzy Search - Full Content",
            "query": {
                "content": "MICHAEL RICHARD LEE, born 2001-02-10, male, postal V3N1H4, mincode 05757079"
            }
        }
    ]
    
    for i, test_case in enumerate(fuzzy_test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        
        result = search_student_by_query(test_case['query'])
        print_search_results(result, max_display=3)

# Example usage
if __name__ == "__main__":
    run_test_suite()