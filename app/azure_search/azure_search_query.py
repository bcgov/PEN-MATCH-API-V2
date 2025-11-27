import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
import difflib

from config.settings import settings

@dataclass
class SearchResult:
    """Structured search result"""
    student_id: str
    pen: Optional[str]
    legal_first_name: Optional[str]
    legal_middle_names: Optional[str]
    legal_last_name: Optional[str]
    dob: Optional[str]
    sex_code: Optional[str]
    postal_code: Optional[str]
    mincode: Optional[str]
    grade: Optional[str]
    local_id: Optional[str]
    search_score: float
    match_type: str = "candidate"  # "perfect" or "candidate"

@dataclass
class SearchResponse:
    """Search response with perfect matches and candidates"""
    perfect_matches: List[SearchResult]
    candidates: List[SearchResult]
    query: Dict[str, Any]
    total_results: int
    search_method: str

class AzureSearchQuery:
    def __init__(self):
        self.search_endpoint = "https://pen-match-api-v2-search.search.windows.net"
        self.index_name = "student-index"
        
        # Azure Search client
        self.credential = DefaultAzureCredential()
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
        
        # OpenAI embedding client
        self.openai_client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version="2023-05-15",
            azure_endpoint=settings.openai_api_base_embedding_3
        )
        
        print("AzureSearchQuery initialized successfully")

    def generate_name_embedding(self, first_name: str, last_name: str, middle_names: str = "") -> List[float]:
        """Generate embedding for name search using same format as import"""
        name_part = f"{first_name} {last_name}".strip()
        text = f"Name: {name_part}, Middlename: {middle_names}"
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding generation error: {e}")
            raise

    def _calculate_field_similarity(self, query_value: str, candidate_value: str, field_type: str = "exact") -> float:
        """Calculate field similarity based on field type"""
        if not query_value or not candidate_value:
            return 0.0
        
        query_clean = str(query_value).strip().upper()
        candidate_clean = str(candidate_value).strip().upper()
        
        if query_clean == candidate_clean:
            return 1.0
        
        if field_type == "postal":
            # Postal code partial matching
            if len(query_clean) >= 3 and len(candidate_clean) >= 3:
                if query_clean[:3] == candidate_clean[:3]:
                    return 0.7
            similarity = difflib.SequenceMatcher(None, query_clean, candidate_clean).ratio()
            return similarity if similarity > 0.6 else 0.0
        
        elif field_type == "mincode":
            # Mincode partial matching
            similarity = difflib.SequenceMatcher(None, query_clean, candidate_clean).ratio()
            return similarity if similarity > 0.8 else 0.0
        
        else:
            # Exact match for other fields
            return 0.0

    def _is_perfect_match(self, query: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
        """Check if candidate is a perfect match for all provided fields"""
        
        # Check all provided fields for exact match
        field_mapping = {
            "pen": "pen",
            "legalFirstName": "legalFirstName", 
            "legalLastName": "legalLastName",
            "legalMiddleNames": "legalMiddleNames",
            "dob": "dob",
            "sexCode": "sexCode",
            "postalCode": "postalCode",
            "mincode": "mincode",
            "grade": "grade",
            "localID": "localID"
        }
        
        for query_field, candidate_field in field_mapping.items():
            query_value = query.get(query_field)
            candidate_value = candidate.get(candidate_field)
            
            if query_value and query_value != 'NULL':
                if not candidate_value or candidate_value == 'NULL':
                    return False
                
                # Special handling for postal and mincode
                if query_field in ["postalCode", "mincode"]:
                    similarity = self._calculate_field_similarity(
                        query_value, candidate_value, 
                        "postal" if query_field == "postalCode" else "mincode"
                    )
                    if similarity < 1.0:  # Must be exact for perfect match
                        return False
                else:
                    # Exact match required
                    if str(query_value).strip().upper() != str(candidate_value).strip().upper():
                        return False
        
        return True

    def _calculate_candidate_score(self, query: Dict[str, Any], candidate: Dict[str, Any], base_score: float) -> float:
        """Calculate candidate score based on field matches"""
        
        score = base_score  # Start with search relevance score
        
        # High priority fields
        if query.get("dob") and candidate.get("dob"):
            if query["dob"] == candidate["dob"]:
                score += 0.3  # DOB exact match bonus
        
        # Bonus points for other fields
        if query.get("sexCode") and candidate.get("sexCode"):
            if query["sexCode"].upper() == candidate["sexCode"].upper():
                score += 0.1
        
        if query.get("postalCode") and candidate.get("postalCode"):
            postal_sim = self._calculate_field_similarity(
                query["postalCode"], candidate["postalCode"], "postal"
            )
            score += postal_sim * 0.15
        
        if query.get("mincode") and candidate.get("mincode"):
            mincode_sim = self._calculate_field_similarity(
                query["mincode"], candidate["mincode"], "mincode"
            )
            score += mincode_sim * 0.15
        
        if query.get("grade") and candidate.get("grade"):
            if query["grade"] == candidate["grade"]:
                score += 0.05
        
        return score

    async def search_students(self, query: Dict[str, Any]) -> SearchResponse:
        """
        Main search function following the specified methodology
        """
        
        # 1. Direct lookup by PEN or Student ID
        if query.get("pen") or query.get("student_id"):
            return await self._direct_lookup(query)
        
        # 2. Hybrid search with name as dominant key
        return await self._hybrid_name_search(query)

    async def _direct_lookup(self, query: Dict[str, Any]) -> SearchResponse:
        """Direct lookup by PEN or Student ID"""
        
        try:
            filter_expr = None
            if query.get("pen"):
                filter_expr = f"pen eq '{query['pen']}'"
            elif query.get("student_id"):
                filter_expr = f"student_id eq '{query['student_id']}'"
            
            results = self.search_client.search(
                search_text="*",
                filter=filter_expr,
                top=1,
                select=[
                    "id", "student_id", "pen", "legalFirstName", "legalMiddleNames", 
                    "legalLastName", "dob", "sexCode", "postalCode", "mincode", 
                    "grade", "localID"
                ]
            )
            
            search_results = []
            for result in results:
                search_result = SearchResult(
                    student_id=result.get('student_id', ''),
                    pen=result.get('pen'),
                    legal_first_name=result.get('legalFirstName'),
                    legal_middle_names=result.get('legalMiddleNames'),
                    legal_last_name=result.get('legalLastName'),
                    dob=result.get('dob'),
                    sex_code=result.get('sexCode'),
                    postal_code=result.get('postalCode'),
                    mincode=result.get('mincode'),
                    grade=result.get('grade'),
                    local_id=result.get('localID'),
                    search_score=1.0,
                    match_type="perfect"
                )
                search_results.append(search_result)
            
            return SearchResponse(
                perfect_matches=search_results,
                candidates=[],
                query=query,
                total_results=len(search_results),
                search_method="direct_lookup"
            )
            
        except Exception as e:
            print(f"Direct lookup error: {e}")
            return SearchResponse([], [], query, 0, "direct_lookup_failed")

    async def _hybrid_name_search(self, query: Dict[str, Any]) -> SearchResponse:
        """Hybrid search using name + other fields"""
        
        # Validate required fields
        if not query.get("legalFirstName") or not query.get("legalLastName"):
            return SearchResponse([], [], query, 0, "missing_required_fields")
        
        try:
            # 1. Generate embedding for vector search
            name_embedding = self.generate_name_embedding(
                query["legalFirstName"], 
                query["legalLastName"], 
                query.get("legalMiddleNames", "")
            )
            
            # 2. Build keyword search text
            keyword_text = f"{query['legalFirstName']} {query.get('legalMiddleNames', '')} {query['legalLastName']}".strip()
            
            # 3. Build hard filters for high priority fields
            filters = []
            
            # DOB as second key - exact match if provided
            if query.get("dob"):
                filters.append(f"dob eq '{query['dob']}'")
            
            # Sex filter if provided
            if query.get("sexCode"):
                filters.append(f"sexCode eq '{query['sexCode']}'")
            
            filter_expression = " and ".join(filters) if filters else None
            
            # 4. Create vector query
            vector_query = VectorizedQuery(
                vector=name_embedding,
                k_nearest_neighbors=100,  # Get more for processing
                fields="nameEmbedding"
            )
            
            # 5. Execute hybrid search
            results = self.search_client.search(
                search_text=keyword_text,
                vector_queries=[vector_query],
                top=100,
                filter=filter_expression,
                select=[
                    "id", "student_id", "pen", "legalFirstName", "legalMiddleNames", 
                    "legalLastName", "dob", "sexCode", "postalCode", "mincode", 
                    "grade", "localID"
                ],
                scoring_profile="identity-ranking" if filter_expression is None else None
            )
            
            # 6. Process results and categorize
            perfect_matches = []
            candidates = []
            
            for result in results:
                candidate_data = {
                    "student_id": result.get('student_id', ''),
                    "pen": result.get('pen'),
                    "legalFirstName": result.get('legalFirstName'),
                    "legalMiddleNames": result.get('legalMiddleNames'),
                    "legalLastName": result.get('legalLastName'),
                    "dob": result.get('dob'),
                    "sexCode": result.get('sexCode'),
                    "postalCode": result.get('postalCode'),
                    "mincode": result.get('mincode'),
                    "grade": result.get('grade'),
                    "localID": result.get('localID')
                }
                
                base_score = result.get('@search.score', 0)
                final_score = self._calculate_candidate_score(query, candidate_data, base_score)
                
                search_result = SearchResult(
                    student_id=candidate_data['student_id'],
                    pen=candidate_data['pen'],
                    legal_first_name=candidate_data['legalFirstName'],
                    legal_middle_names=candidate_data['legalMiddleNames'],
                    legal_last_name=candidate_data['legalLastName'],
                    dob=candidate_data['dob'],
                    sex_code=candidate_data['sexCode'],
                    postal_code=candidate_data['postalCode'],
                    mincode=candidate_data['mincode'],
                    grade=candidate_data['grade'],
                    local_id=candidate_data['localID'],
                    search_score=final_score
                )
                
                # Check if perfect match
                if self._is_perfect_match(query, candidate_data):
                    search_result.match_type = "perfect"
                    perfect_matches.append(search_result)
                else:
                    search_result.match_type = "candidate"
                    candidates.append(search_result)
            
            # 7. Sort by score
            perfect_matches.sort(key=lambda x: x.search_score, reverse=True)
            candidates.sort(key=lambda x: x.search_score, reverse=True)
            
            # 8. Apply business rules
            if len(perfect_matches) == 1 and len(candidates) == 0:
                # Single perfect match - return as perfect match
                return SearchResponse(
                    perfect_matches=perfect_matches,
                    candidates=[],
                    query=query,
                    total_results=1,
                    search_method="single_perfect_match"
                )
            else:
                # Multiple candidates or imperfect matches - return as candidates
                all_candidates = perfect_matches + candidates
                all_candidates.sort(key=lambda x: x.search_score, reverse=True)
                
                return SearchResponse(
                    perfect_matches=[],
                    candidates=all_candidates[:20],  # Limit to top 20
                    query=query,
                    total_results=len(all_candidates),
                    search_method="multiple_candidates"
                )
            
        except Exception as e:
            print(f"Hybrid search error: {e}")
            return SearchResponse([], [], query, 0, "hybrid_search_failed")

    def print_search_response(self, response: SearchResponse):
        """Pretty print search response"""
        print(f"\n=== SEARCH RESULTS ===")
        print(f"Search Method: {response.search_method}")
        print(f"Total Results: {response.total_results}")
        print(f"Perfect Matches: {len(response.perfect_matches)}")
        print(f"Candidates: {len(response.candidates)}")
        
        if response.perfect_matches:
            print(f"\n--- PERFECT MATCHES ({len(response.perfect_matches)}) ---")
            for i, result in enumerate(response.perfect_matches, 1):
                self._print_result(result, i)
        
        if response.candidates:
            print(f"\n--- CANDIDATES ({len(response.candidates)}) ---")
            for i, result in enumerate(response.candidates, 1):
                self._print_result(result, i)
    
    def _print_result(self, result: SearchResult, index: int):
        """Print individual result"""
        print(f"\n{index}. {result.legal_first_name} {result.legal_middle_names or ''} {result.legal_last_name}".strip())
        print(f"   Student ID: {result.student_id}")
        print(f"   PEN: {result.pen}")
        print(f"   DOB: {result.dob}")
        print(f"   Sex: {result.sex_code}")
        print(f"   Postal: {result.postal_code}")
        print(f"   Mincode: {result.mincode}")
        print(f"   Score: {result.search_score:.4f} ({result.match_type})")


# Test functions
async def run_test_suite():
    """Run comprehensive test suite"""
    search_service = AzureSearchQuery()
    
    # Base query
    base_query = {
        "legalFirstName": "MICHAEL",
        "legalLastName": "LEE",
    }
    
    test_cases = [
        # Correct Query
        ("First Name + Last Name", base_query),
        ("First Name + Last Name + Middle", {**base_query, "legalMiddleNames": "RICHARD"}),
        ("First Name + Last Name + Postal", {**base_query, "postalCode": "V3N1H4"}),
        ("First Name + Last Name + Mincode", {**base_query, "mincode": "05757079"}),
        ("First Name + Last Name + DOB", {**base_query, "dob": "2001-02-10"}),
        ("First Name + Last Name + Sex", {**base_query, "sexCode": "M"}),
        ("First Name + Last Name + ALL", {
            **base_query, 
            "legalMiddleNames": "RICHARD",
            "dob": "2001-02-10",
            "sexCode": "M",
            "postalCode": "V3N1H4",
            "mincode": "05757079"
        }),
        
        # Typo Query
        ("Typo: MICHAL LE + Middle RICHAR", {
            "legalFirstName": "MICHAL",
            "legalLastName": "LE",
            "legalMiddleNames": "RICHAR"
        }),
        ("Typo: Postal V3N4H1", {**base_query, "postalCode": "V3N4H1"}),
        ("Typo: Mincode 057079", {**base_query, "mincode": "057079"}),
        
        # Direct lookup
        ("Direct PEN Lookup", {"pen": "124809765"}),
    ]
    
    for test_name, query in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        response = await search_service.search_students(query)
        search_service.print_search_response(response)


if __name__ == "__main__":
    asyncio.run(run_test_suite())