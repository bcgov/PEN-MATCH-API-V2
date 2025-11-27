import asyncio 
import json
import time
import os
from datetime import datetime
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
    match_type: str = "candidate"

@dataclass
class SearchResponse:
    """Search response with perfect matches and candidates"""
    perfect_matches: List[SearchResult]
    candidates: List[SearchResult]
    query: Dict[str, Any]
    total_results: int
    search_method: str
    search_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    azure_search_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    truncated_at_100: bool = False

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
        
        # Setup logging
        self._setup_logging()
        
        # Field matching thresholds
        self.thresholds = {
            # We no longer use @search.score as a hard cutoff
            "name_embedding_min": 0.0,   # <<< CHANGE (not used for filtering anymore)
            "postal_max_diff": 2,
            "mincode_max_diff": 3,
            "name_similarity_min": 0.6
        }
        
        print("AzureSearchQuery initialized successfully")

    def _setup_logging(self):
        """Setup debug logging to file"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'log')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"azure_search_debug_{timestamp}.log")
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Azure Search Debug Log - {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

    def _log_debug(self, message: str):
        """Log debug message to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")

    def generate_name_embedding(self, first_name: str, last_name: str, middle_names: str = "") -> Tuple[List[float], float]:
        """Generate embedding for name search"""
        start_time = time.perf_counter()
        
        # Format: Name: FIRST MIDDLE LAST.
        name_parts = []
        if first_name.strip():
            name_parts.append(first_name.strip())
        if middle_names.strip():
            name_parts.append(middle_names.strip())
        if last_name.strip():
            name_parts.append(last_name.strip())
        
        full_name = ' '.join(name_parts)
        text = f"{full_name}."
        
        self._log_debug(f"Generating embedding for: {text}")
        self._log_debug(f"Input breakdown - First: '{first_name}', Last: '{last_name}', Middle: '{middle_names}'")
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            embedding_time = (time.perf_counter() - start_time) * 1000
            self._log_debug(f"Embedding generated in {embedding_time:.2f}ms")
            self._log_debug(f"First 5 embedding values: {response.data[0].embedding[:5]}")
            return response.data[0].embedding, embedding_time
        except Exception as e:
            embedding_time = (time.perf_counter() - start_time) * 1000
            self._log_debug(f"Embedding error after {embedding_time:.2f}ms: {e}")
            raise

    def _calculate_field_similarity(self, query_value: str, candidate_value: str, field_type: str = "exact") -> float:
        """Calculate field similarity with thresholds"""
        if not query_value or not candidate_value:
            return 0.0
        
        query_clean = str(query_value).strip().upper()
        candidate_clean = str(candidate_value).strip().upper()
        
        if query_clean == candidate_clean:
            return 1.0
        
        if field_type == "postal":
            # Postal code: max 2 character differences
            diff_count = sum(1 for a, b in zip(query_clean, candidate_clean) if a != b)
            diff_count += abs(len(query_clean) - len(candidate_clean))
            return 0.0 if diff_count > self.thresholds["postal_max_diff"] else 0.7
        
        elif field_type == "mincode":
            # Mincode: max 3 character differences
            diff_count = sum(1 for a, b in zip(query_clean, candidate_clean) if a != b)
            diff_count += abs(len(query_clean) - len(candidate_clean))
            return 0.0 if diff_count > self.thresholds["mincode_max_diff"] else 0.8
        
        elif field_type == "name":
            # Name fields: minimum similarity threshold
            similarity = difflib.SequenceMatcher(None, query_clean, candidate_clean).ratio()
            return similarity if similarity >= self.thresholds["name_similarity_min"] else 0.0
        
        else:
            return 0.0

    def _count_exact_field_matches(self, query: Dict[str, Any], candidate: Dict[str, Any]) -> int:
        """Count exactly matching fields (excluding names)"""
        exact_matches = 0
        check_fields = ["dob", "sexCode", "postalCode", "mincode", "grade", "localID"]
        
        for field in check_fields:
            query_value = query.get(field)
            candidate_value = candidate.get(field)
            
            if (query_value and query_value != 'NULL' and 
                candidate_value and candidate_value != 'NULL'):
                if str(query_value).strip().upper() == str(candidate_value).strip().upper():
                    exact_matches += 1
        
        return exact_matches

    def _is_perfect_match(self, query: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
        """Check if candidate perfectly matches all provided fields"""
        field_mapping = {
            "pen": "pen", "legalFirstName": "legalFirstName", 
            "legalLastName": "legalLastName", "legalMiddleNames": "legalMiddleNames",
            "dob": "dob", "sexCode": "sexCode", "postalCode": "postalCode",
            "mincode": "mincode", "grade": "grade", "localID": "localID"
        }
        
        for query_field, candidate_field in field_mapping.items():
            query_value = query.get(query_field)
            candidate_value = candidate.get(candidate_field)
            
            if query_value and query_value != 'NULL':
                if not candidate_value or candidate_value == 'NULL':
                    return False
                if str(query_value).strip().upper() != str(candidate_value).strip().upper():
                    return False
        
        return True

    def _is_reasonable_candidate(self, query: Dict[str, Any], candidate: Dict[str, Any], base_score: float) -> bool:
        """
        Check if candidate meets threshold criteria.

        IMPORTANT CHANGE:
        - We no longer reject candidates purely based on @search.score.
        - We only drop candidates that clearly violate strong fields like postal/mincode.
        """
        # <<< CHANGE: removed name_embedding_min score cutoff
        self._log_debug(
            f"Evaluating candidate {candidate.get('legalFirstName')} "
            f"{candidate.get('legalLastName')} with score={base_score}"
        )

        # Field-specific thresholds (only if both sides present)
        for field, field_type in [("postalCode", "postal"), ("mincode", "mincode")]:
            query_value = query.get(field)
            candidate_value = candidate.get(field)
            
            if (query_value and query_value != 'NULL' and 
                candidate_value and candidate_value != 'NULL'):
                similarity = self._calculate_field_similarity(query_value, candidate_value, field_type)
                if similarity == 0.0:  # Below threshold → clearly wrong candidate
                    self._log_debug(
                        f"Rejected by {field} similarity. "
                        f"Query='{query_value}', Candidate='{candidate_value}'"
                    )
                    return False
        
        # If we reach here, treat as reasonable
        return True  # <<< CHANGE (always accept if not obviously wrong)

    async def search_students(self, query: Dict[str, Any]) -> SearchResponse:
        """Main search function"""
        start_time = time.perf_counter()
        self._log_debug(f"=== SEARCH STARTED ===")
        self._log_debug(f"Query: {json.dumps(query, indent=2)}")
        
        # Direct lookup by PEN or Student ID
        if query.get("pen") or query.get("student_id"):
            response = await self._direct_lookup(query)
        else:
            response = await self._hybrid_name_search(query)
        
        total_time = (time.perf_counter() - start_time) * 1000
        response.search_time_ms = total_time
        
        self._log_debug(f"=== SEARCH COMPLETED ===")
        self._log_debug(f"Total time: {total_time:.2f}ms")
        self._log_debug(f"Results: {response.total_results} total, {len(response.perfect_matches)} perfect, {len(response.candidates)} candidates")
        self._log_debug("")
        
        return response

    async def _direct_lookup(self, query: Dict[str, Any]) -> SearchResponse:
        """Direct lookup by PEN or Student ID"""
        self._log_debug("Using direct lookup method")
        
        try:
            filter_expr = f"pen eq '{query['pen']}'" if query.get("pen") else f"student_id eq '{query['student_id']}'"
            
            azure_start = time.perf_counter()
            results = self.search_client.search(
                search_text="*", filter=filter_expr, top=1,
                select=["id", "student_id", "pen", "legalFirstName", "legalMiddleNames", 
                       "legalLastName", "dob", "sexCode", "postalCode", "mincode", "grade", "localID"]
            )
            azure_time = (time.perf_counter() - azure_start) * 1000
            
            process_start = time.perf_counter()
            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    student_id=result.get('student_id', ''), pen=result.get('pen'),
                    legal_first_name=result.get('legalFirstName'), 
                    legal_middle_names=result.get('legalMiddleNames'),
                    legal_last_name=result.get('legalLastName'), dob=result.get('dob'),
                    sex_code=result.get('sexCode'), postal_code=result.get('postalCode'),
                    mincode=result.get('mincode'), grade=result.get('grade'),
                    local_id=result.get('localID'), search_score=1.0, match_type="perfect"
                ))
            process_time = (time.perf_counter() - process_start) * 1000
            
            return SearchResponse(
                perfect_matches=search_results, candidates=[], query=query,
                total_results=len(search_results), search_method="direct_lookup",
                embedding_time_ms=0.0, azure_search_time_ms=azure_time, processing_time_ms=process_time
            )
            
        except Exception as e:
            self._log_debug(f"Direct lookup error: {e}")
            return SearchResponse([], [], query, 0, "direct_lookup_failed")

    async def _hybrid_name_search(self, query: Dict[str, Any]) -> SearchResponse:
        """Hybrid search with improved candidate logic"""
        if not query.get("legalFirstName") or not query.get("legalLastName"):
            return SearchResponse([], [], query, 0, "missing_required_fields")
        
        self._log_debug("Using hybrid name search method")
        
        try:
            # Generate embedding
            name_embedding, embedding_time = self.generate_name_embedding(
                query["legalFirstName"], query["legalLastName"], query.get("legalMiddleNames", "")
            )
            
            # Build search parameters
            keyword_text = f"{query['legalFirstName']} {query.get('legalMiddleNames', '')} {query['legalLastName']}".strip()
            
            # Only use exact filters for high-confidence fields
            filters = []
            if query.get("dob"):
                filters.append(f"dob eq '{query['dob']}'")
            filter_expression = " and ".join(filters) if filters else None
            
            # Execute search
            azure_start = time.perf_counter()
            vector_query = VectorizedQuery(vector=name_embedding, k_nearest_neighbors=1000, fields="nameEmbedding")
            
            results = self.search_client.search(
                search_text=keyword_text, vector_queries=[vector_query], top=1000,
                filter=filter_expression,
                select=["id", "student_id", "pen", "legalFirstName", "legalMiddleNames", 
                       "legalLastName", "dob", "sexCode", "postalCode", "mincode", "grade", "localID"],
                scoring_profile="identity-ranking" if filter_expression is None else None
            )
            azure_time = (time.perf_counter() - azure_start) * 1000
            
            # Process results with improved logic
            process_start = time.perf_counter()
            perfect_matches = []
            candidates = []
            two_field_exact_matches = []
            
            total_seen = 0  # <<< optional debug counter

            for result in results:
                total_seen += 1
                candidate_data = {
                    "student_id": result.get('student_id', ''), "pen": result.get('pen'),
                    "legalFirstName": result.get('legalFirstName'), 
                    "legalMiddleNames": result.get('legalMiddleNames'),
                    "legalLastName": result.get('legalLastName'), "dob": result.get('dob'),
                    "sexCode": result.get('sexCode'), "postalCode": result.get('postalCode'),
                    "mincode": result.get('mincode'), "grade": result.get('grade'), 
                    "localID": result.get('localID')
                }
                
                base_score = result.get('@search.score', 0)
                self._log_debug(
                    f"Raw result {total_seen}: "
                    f"{candidate_data['legalFirstName']} {candidate_data['legalLastName']} "
                    f"score={base_score}"
                )
                
                # Check if reasonable candidate first
                if not self._is_reasonable_candidate(query, candidate_data, base_score):
                    continue
                
                search_result = SearchResult(
                    student_id=candidate_data['student_id'], pen=candidate_data['pen'],
                    legal_first_name=candidate_data['legalFirstName'], 
                    legal_middle_names=candidate_data['legalMiddleNames'],
                    legal_last_name=candidate_data['legalLastName'], dob=candidate_data['dob'],
                    sex_code=candidate_data['sexCode'], postal_code=candidate_data['postalCode'],
                    mincode=candidate_data['mincode'], grade=candidate_data['grade'],
                    local_id=candidate_data['localID'], search_score=base_score
                )
                
                # Categorize results
                if self._is_perfect_match(query, candidate_data):
                    search_result.match_type = "perfect"
                    perfect_matches.append(search_result)
                else:
                    exact_field_count = self._count_exact_field_matches(query, candidate_data)
                    if exact_field_count >= 2:
                        # Two fields exactly match
                        search_result.match_type = "two_field_exact"
                        two_field_exact_matches.append(search_result)
                    else:
                        search_result.match_type = "candidate"
                        candidates.append(search_result)
            
            process_time = (time.perf_counter() - process_start) * 1000
            
            # Apply business rules
            final_perfect = perfect_matches
            final_candidates = candidates
            
            # Two field exact match logic
            if two_field_exact_matches:
                if len(two_field_exact_matches) == 1:
                    # Single two-field match becomes perfect match
                    two_field_exact_matches[0].match_type = "perfect"
                    final_perfect.extend(two_field_exact_matches)
                else:
                    # Multiple two-field matches become candidates, exclude other candidates
                    for match in two_field_exact_matches:
                        match.match_type = "candidate"
                    final_candidates = two_field_exact_matches
            
            # Sort results
            final_perfect.sort(key=lambda x: x.search_score, reverse=True)
            final_candidates.sort(key=lambda x: x.search_score, reverse=True)
            
            # Handle over 100 candidates
            truncated = len(final_candidates) > 100
            total_before_truncation = len(final_perfect) + len(final_candidates)
            
            if truncated:
                self._log_debug(f"Truncating {len(final_candidates)} candidates to 100")
                final_candidates = final_candidates[:100]
            
            # Determine search method
            if len(final_perfect) == 1 and len(final_candidates) == 0:
                search_method = "single_perfect_match"
            elif len(final_perfect) > 0:
                search_method = "multiple_perfect_matches"
            else:
                search_method = "candidates_only"
                if truncated:
                    search_method += "_truncated"
            
            self._log_debug(
                f"Total raw results from Azure: {total_seen}, "
                f"kept={len(final_perfect)+len(final_candidates)}"
            )

            return SearchResponse(
                perfect_matches=final_perfect, candidates=final_candidates, query=query,
                total_results=total_before_truncation, search_method=search_method,
                embedding_time_ms=embedding_time, azure_search_time_ms=azure_time, 
                processing_time_ms=process_time, truncated_at_100=truncated
            )
            
        except Exception as e:
            self._log_debug(f"Hybrid search error: {e}")
            return SearchResponse([], [], query, 0, "hybrid_search_failed")

    def print_search_response(self, response: SearchResponse, debug_limit: int = 5):
        """Print search response with debug limit"""
        print(f"\n=== SEARCH RESULTS ===")
        print(f"Search Method: {response.search_method}")
        print(f"Total Results: {response.total_results}")
        print(f"Perfect Matches: {len(response.perfect_matches)}")
        print(f"Candidates: {len(response.candidates)}")
        if response.truncated_at_100:
            print(f"NOTICE: Over 100 candidates found, showing first 100")
        print(f"TIMING:")
        print(f"   Total: {response.search_time_ms:.2f}ms")
        print(f"   Embedding: {response.embedding_time_ms:.2f}ms") 
        print(f"   Azure Search: {response.azure_search_time_ms:.2f}ms")
        print(f"   Processing: {response.processing_time_ms:.2f}ms")
        print(f"Debug Log: {self.log_file}")
        
        if response.perfect_matches:
            print(f"\n--- PERFECT MATCHES ({len(response.perfect_matches)}) ---")
            for i, result in enumerate(response.perfect_matches[:debug_limit], 1):
                self._print_result(result, i)
            if len(response.perfect_matches) > debug_limit:
                print(f"... and {len(response.perfect_matches) - debug_limit} more perfect matches")
        
        if response.candidates:
            print(f"\n--- CANDIDATES ({len(response.candidates)} showing, top {debug_limit}) ---")
            for i, result in enumerate(response.candidates[:debug_limit], 1):
                self._print_result(result, i)
            if len(response.candidates) > debug_limit:
                print(f"... and {len(response.candidates) - debug_limit} more candidates")
    
    def _print_result(self, result: SearchResult, index: int):
        """Print individual result"""
        print(f"\n{index}. {result.legal_first_name} {result.legal_middle_names or ''} {result.legal_last_name}".strip())
        print(f"   Student ID: {result.student_id} | PEN: {result.pen}")
        print(f"   DOB: {result.dob} | Sex: {result.sex_code}")
        print(f"   Postal: {result.postal_code} | Mincode: {result.mincode}")
        print(f"   Score: {result.search_score:.4f} ({result.match_type})")


async def run_test_suite():
    """Run test suite with single-field typo cases"""
    search_service = AzureSearchQuery()
    
    base_query = {"legalFirstName": "MICHAEL", "legalLastName": "LEE"}
    
    test_cases = [
        # Correct queries
        ("Name Only", base_query),
        ("Name + Middle", {**base_query, "legalMiddleNames": "RICHARD"}),
        ("Name + DOB", {**base_query, "dob": "2001-02-10"}),
        ("Name + Sex", {**base_query, "sexCode": "M"}),
        ("Name + All Fields", {**base_query, "legalMiddleNames": "RICHARD", "dob": "2001-02-10", 
                              "sexCode": "M", "postalCode": "V3N1H4", "mincode": "05757079"}),
        
        # Single field typos
        ("Typo: First Name", {"legalFirstName": "MICHAL", "legalLastName": "LEE"}),
        ("Typo: Last Name", {"legalFirstName": "MICHAEL", "legalLastName": "LE"}),
        ("Typo: Postal Code", {**base_query, "postalCode": "V3N4H1"}),
        ("Typo: Mincode", {**base_query, "mincode": "05757999"}),
        
        # Direct lookup
        ("Direct PEN", {"pen": "124809765"}),
    ]
    
    print(f"Running Azure Search Test Suite")
    print(f"Debug logs: app/log/")
    
    for test_name, query in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        response = await search_service.search_students(query)
        search_service.print_search_response(response, debug_limit=5)


if __name__ == "__main__":
    asyncio.run(run_test_suite())
