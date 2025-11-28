from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.identity import DefaultAzureCredential
import os
import difflib
from typing import Dict, List, Any
from openai import AzureOpenAI

from config.settings import settings

class StudentSearchService:
    def __init__(self):
        self.search_endpoint = "https://pen-match-api-v2-search.search.windows.net"
        self.index_name = "student-index"
        
        # Use DefaultAzureCredential for authentication
        self.credential = DefaultAzureCredential()
        
        # OpenAI embedding client - same as import
        self.openai_client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version="2023-05-15",
            azure_endpoint=settings.openai_api_base_embedding_3
        )
        
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
    
    def generate_embedding(self, query_data: Dict[str, Any]) -> List[float]:
        """Generate embedding using text-embedding-3-large model - same as import"""
        # Format: Name: FIRST MIDDLE LAST.
        first_name = query_data.get('legalFirstName', '').strip()
        last_name = query_data.get('legalLastName', '').strip()
        middle_names = query_data.get('legalMiddleNames', '').strip()
        
        if first_name == 'NULL':
            first_name = ''
        if last_name == 'NULL':
            last_name = ''
        if middle_names == 'NULL':
            middle_names = ''
        
        # Build full name with middle names in between
        name_parts = []
        if first_name:
            name_parts.append(first_name)
        if middle_names:
            name_parts.append(middle_names)
        if last_name:
            name_parts.append(last_name)
        
        full_name = ' '.join(name_parts)
        text = f"{full_name}."
        
        if not text.strip() or text == ".":
            return None
        
        print(f"Generating embedding for: {text}")
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                dimensions=3072
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"Error generating embedding for '{text}': {e}")
            return None
    
    def _calculate_postal_similarity(self, query_postal: str, candidate_postal: str) -> float:
        """Calculate postal code similarity with improved matching"""
        if not query_postal or not candidate_postal:
            return 0.0
        
        query_clean = query_postal.replace(" ", "").upper()
        candidate_clean = candidate_postal.replace(" ", "").upper()
        
        if query_clean == candidate_clean:
            return 1.0
        
        # Check if first 3 characters match (same area)
        if len(query_clean) >= 3 and len(candidate_clean) >= 3:
            if query_clean[:3] == candidate_clean[:3]:
                return 0.7  # Partial match for same area
        
        # Fuzzy matching for close postal codes
        similarity = difflib.SequenceMatcher(None, query_clean, candidate_clean).ratio()
        return similarity if similarity > 0.5 else 0.0
    
    def _calculate_mincode_similarity(self, query_mincode: str, candidate_mincode: str) -> float:
        """Calculate mincode similarity (exact match preferred)"""
        if not query_mincode or not candidate_mincode:
            return 0.0
        
        query_clean = str(query_mincode).strip()
        candidate_clean = str(candidate_mincode).strip()
        
        if query_clean == candidate_clean:
            return 1.0
        
        # Partial matching for mincode
        similarity = difflib.SequenceMatcher(None, query_clean, candidate_clean).ratio()
        return similarity if similarity > 0.8 else 0.0
    
    def _calculate_sex_similarity(self, query_sex: str, candidate_sex: str) -> float:
        """Calculate sex similarity (exact match only)"""
        if not query_sex or not candidate_sex:
            return 0.0
        
        return 1.0 if query_sex.upper() == candidate_sex.upper() else 0.0
    
    def _calculate_dob_similarity(self, query_dob: str, candidate_dob: str) -> float:
        """Calculate DOB similarity (exact match only)"""
        if not query_dob or not candidate_dob:
            return 0.0
        
        return 1.0 if query_dob == candidate_dob else 0.0
    
    def _has_middle_name_query(self, query: Dict[str, Any]) -> bool:
        """Check if query has middle name"""
        middle = query.get("legalMiddleNames", "")
        return middle and middle.strip() and middle != 'NULL'
    
    def _is_reasonable_candidate(self, query: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
        """Filter for reasonable candidates based on basic criteria"""
        
        # Sex filter - if query has sex, candidate must match
        query_sex = query.get("sexCode", "").upper()
        candidate_sex = candidate.get("sexCode", "").upper()
        if query_sex and candidate_sex and query_sex != candidate_sex:
            return False
        
        # Base search score threshold - Azure Search similarity must be reasonable
        base_score = candidate.get("@search.score", 0)
        if base_score < 0.5:  # Minimum Azure Search relevance
            return False
        
        return True
    
    def _calculate_soft_score(self, query: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """Calculate soft scoring for postal, mincode, sex, dob with adaptive weights"""
        soft_score = 0.0
        
        # Check if query has middle name - if not, increase weights for other fields
        has_middle_name = self._has_middle_name_query(query)
        
        if has_middle_name:
            # Normal weights when middle name is provided
            postal_weight = 0.15    
            mincode_weight = 0.20   
            sex_weight = 0.10       
            dob_weight = 0.25       
        else:
            # Higher weights when middle name is missing - compensate for name difference
            postal_weight = 0.25    # Much higher weight
            mincode_weight = 0.30   # Much higher weight
            sex_weight = 0.15       # Much higher weight
            dob_weight = 0.35       # Much higher weight
        
        # DOB similarity (highest priority)
        dob_sim = self._calculate_dob_similarity(
            query.get("dob", ""),
            candidate.get("dob", "")
        )
        soft_score += dob_sim * dob_weight
        
        # Postal code similarity
        postal_sim = self._calculate_postal_similarity(
            query.get("postalCode", ""),
            candidate.get("postalCode", "")
        )
        soft_score += postal_sim * postal_weight
        
        # Mincode similarity  
        mincode_sim = self._calculate_mincode_similarity(
            query.get("mincode", ""),
            candidate.get("mincode", "")
        )
        soft_score += mincode_sim * mincode_weight
        
        # Sex similarity
        sex_sim = self._calculate_sex_similarity(
            query.get("sexCode", ""),
            candidate.get("sexCode", "")
        )
        soft_score += sex_sim * sex_weight
        
        return soft_score
    
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
            # Step 2: Enhanced fuzzy search with field scoring + vector search
            fuzzy_results = self._enhanced_fuzzy_search(query_data)
            return {
                "status": "success",
                "results": fuzzy_results["results"],
                "count": fuzzy_results["count"],
                "search_type": "fuzzy_match",
                "methodology": fuzzy_results["methodology"]
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
    
    def _enhanced_fuzzy_search(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced fuzzy search with vector embedding + field-based scoring
        """
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query_data)
        
        # Build search text from name fields for text search fallback
        search_terms = []
        if query_data.get("legalFirstName"):
            search_terms.append(query_data["legalFirstName"])
        if query_data.get("legalLastName"):
            search_terms.append(query_data["legalLastName"])
        if query_data.get("legalMiddleNames"):
            search_terms.append(query_data["legalMiddleNames"])
        
        search_text = " ".join(search_terms) if search_terms else query_data.get("content", "")
        
        if not search_text and not query_embedding:
            return {"results": [], "count": 0, "methodology": "No search text or embedding available"}
        
        try:
            # Prepare vector query if embedding is available
            vector_queries = []
            if query_embedding:
                vector_query = VectorizedQuery(
                    vector=query_embedding,
                    k_nearest_neighbors=500,  # Get 500 nearest neighbors
                    fields="nameEmbedding"
                )
                vector_queries.append(vector_query)
            
            # Perform hybrid search (vector + text)
            if vector_queries and search_text:
                # Hybrid search: both vector and text
                results = self.search_client.search(
                    search_text=search_text,
                    vector_queries=vector_queries,
                    search_fields=["content", "legalFirstName", "legalLastName", "legalMiddleNames"],
                    scoring_profile="identityScoring",
                    top=500,
                    include_total_count=True
                )
                search_method = "hybrid_vector_text"
            elif vector_queries:
                # Vector-only search
                results = self.search_client.search(
                    search_text="*",
                    vector_queries=vector_queries,
                    top=500,
                    include_total_count=True
                )
                search_method = "vector_only"
            else:
                # Text-only search (fallback)
                results = self.search_client.search(
                    search_text=search_text,
                    search_fields=["content", "legalFirstName", "legalLastName", "legalMiddleNames"],
                    scoring_profile="identityScoring",
                    top=500,
                    include_total_count=True
                )
                search_method = "text_only"
            
            candidates_list = list(results)
            print(f"Azure Search returned {len(candidates_list)} initial candidates using {search_method}")
            
            # Apply reasonable filtering and soft scoring
            has_middle_name = self._has_middle_name_query(query_data)
            scored_candidates = []
            
            for candidate in candidates_list:
                # Apply reasonable candidate filter
                if not self._is_reasonable_candidate(query_data, candidate):
                    continue
                
                # Calculate soft score for additional fields
                soft_score = self._calculate_soft_score(query_data, candidate)
                
                # Base score from Azure Search
                base_score = candidate.get("@search.score", 0)
                
                # Final score = Azure Search relevance + field similarity bonuses
                final_score = base_score + soft_score
                
                # Add scoring details to candidate
                candidate["base_search_score"] = base_score
                candidate["soft_score"] = soft_score
                candidate["final_score"] = final_score
                candidate["has_middle_name_query"] = has_middle_name
                candidate["search_method"] = search_method
                
                scored_candidates.append(candidate)
            
            # Sort by final score and limit to top 100
            scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
            top_candidates = scored_candidates[:100]
            
            # Debug: Print top 5 candidates
            print(f"\n=== DEBUG: TOP 5 FUZZY SEARCH CANDIDATES ({search_method.upper()}) ===")
            for i, candidate in enumerate(top_candidates[:5], 1):
                middle = candidate.get('legalMiddleNames', '') or ''
                print(f"{i}. {candidate.get('legalFirstName', '')} {middle} {candidate.get('legalLastName', '')}")
                print(f"   Sex: {candidate.get('sexCode', '')}, DOB: {candidate.get('dob', '')}")
                print(f"   Postal: {candidate.get('postalCode', '')}, Mincode: {candidate.get('mincode', '')}")
                print(f"   Base Score: {candidate.get('base_search_score', 0):.4f}, Soft Score: {candidate.get('soft_score', 0):.4f}")
                print(f"   Final Score: {candidate.get('final_score', 0):.4f}")
                print()
            
            methodology = {
                "step1": f"Azure Search with {search_method} (embedding + name fields)",
                "step2": f"Reasonable filtering (sex match, min score 0.5)",
                "step3": f"Soft scoring (DOB, postal, mincode, sex) - weights adjusted for middle name: {has_middle_name}",
                "step4": "Final ranking (base score + soft score)",
                "search_method": search_method,
                "embedding_generated": query_embedding is not None,
                "candidates_before_filtering": len(candidates_list),
                "candidates_after_filtering": len(scored_candidates),
                "has_middle_name_in_query": has_middle_name
            }
            
            return {
                "results": top_candidates,
                "count": len(top_candidates),
                "methodology": methodology
            }
            
        except Exception as e:
            print(f"Error in enhanced fuzzy search: {str(e)}")
            return {"results": [], "count": 0, "methodology": f"Error: {str(e)}"}

def search_student_by_query(query_data: Dict[str, str]) -> Dict[str, Any]:
    """
    Main function to search for students
    """
    search_service = StudentSearchService()
    return search_service.search_students(query_data)

def print_search_results(result: Dict[str, Any], max_display: int = 5):
    """
    Print search results in a formatted way (updated to show top 5 for debug)
    """
    print(f"\n=== SEARCH RESULTS ===")
    print(f"Status: {result['status']}")
    print(f"Search Type: {result['search_type'].upper()}")
    print(f"Total Count: {result['count']}")
    
    # Print methodology for fuzzy search
    if result['search_type'] == 'fuzzy_match' and 'methodology' in result:
        methodology = result['methodology']
        if isinstance(methodology, dict):
            print(f"\nMethodology:")
            print(f"  - Step 1: {methodology.get('step1', 'N/A')}")
            print(f"  - Step 2: {methodology.get('step2', 'N/A')}")
            print(f"  - Step 3: {methodology.get('step3', 'N/A')}")
            print(f"  - Step 4: {methodology.get('step4', 'N/A')}")
            print(f"  - Search Method: {methodology.get('search_method', 'N/A')}")
            print(f"  - Embedding Generated: {methodology.get('embedding_generated', False)}")
            print(f"  - Candidates before/after filtering: {methodology.get('candidates_before_filtering', 0)}/{methodology.get('candidates_after_filtering', 0)}")
    
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
            print(f"   Base Score: {candidate.get('base_search_score', 'N/A')}")
            print(f"   Soft Score: {candidate.get('soft_score', 'N/A')}")
            print(f"   Final Score: {candidate.get('final_score', 'N/A')}")
            print(f"   Search Method: {candidate.get('search_method', 'N/A')}")

# Test cases
def run_test_suite():
    """
    Run test suite with exact and fuzzy search tests
    """
    print("="*60)
    print("AZURE SEARCH TEST SUITE (WITH VECTOR EMBEDDING)")
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
        }
    ]
    
    for i, test_case in enumerate(exact_test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        
        result = search_student_by_query(test_case['query'])
        print_search_results(result, max_display=5)
    
    # FUZZY MATCH TESTS
    print("\n\nFUZZY MATCH TESTS (WITH VECTOR EMBEDDING)")
    print("="*40)
    
    fuzzy_test_cases = [
        {
            "name": "Fuzzy Search - Similar Name",
            "query": {
                "legalFirstName": "MICHEAL",  # Typo
                "legalLastName": "LEE"
            }
        },
        {
            "name": "Fuzzy Search - Name with DOB",
            "query": {
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE",
                "dob": "2001-02-10"
            }
        },
        {
            "name": "Fuzzy Search - Name with Postal",
            "query": {
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE",
                "postalCode": "V3N1H4"
            }
        },
        {
            "name": "Fuzzy Search - Common Name",
            "query": {
                "legalFirstName": "DAVID",
                "legalLastName": "WANG",
                "sexCode": "M"
            }
        },
        {
            "name": "Fuzzy Search - Full Info",
            "query": {
                "legalFirstName": "MICHAEL",
                "legalLastName": "LEE",
                "legalMiddleNames": "RICHARD",
                "dob": "2001-02-10",
                "sexCode": "M",
                "postalCode": "V3N1H4",
                "mincode": "05757079"
            }
        }
    ]
    
    for i, test_case in enumerate(fuzzy_test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        
        result = search_student_by_query(test_case['query'])
        print_search_results(result, max_display=5)

# Example usage
if __name__ == "__main__":
    run_test_suite()