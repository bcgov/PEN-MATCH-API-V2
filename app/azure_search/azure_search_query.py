import asyncio
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

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
    content: Optional[str]
    search_score: float
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    metadata_boost: Optional[float] = None

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

    async def hybrid_search(self, 
                           first_name: str,
                           last_name: str,
                           middle_names: str = "",
                           dob: Optional[str] = None,
                           sex_code: Optional[str] = None,
                           postal_code: Optional[str] = None,
                           mincode: Optional[str] = None,
                           top: int = 20,
                           vector_weight: float = 0.6,
                           keyword_weight: float = 0.3,
                           metadata_weight: float = 0.1) -> List[SearchResult]:
        """
        Hybrid search combining vector, keyword, and metadata filtering
        
        Strategy:
        - 60% vector similarity (semantic name matching)
        - 30% keyword score (lexical fallback) 
        - 10% metadata match (DOB, postal, etc.)
        """
        
        # 1. Generate embedding for vector search
        name_embedding = self.generate_name_embedding(first_name, last_name, middle_names)
        
        # 2. Build keyword search text
        keyword_text = f"{first_name} {middle_names} {last_name}".strip()
        
        # 3. Build filters for metadata
        filters = []
        if dob:
            filters.append(f"dob eq '{dob}'")
        if sex_code:
            filters.append(f"sexCode eq '{sex_code}'")
        if postal_code:
            filters.append(f"postalCode eq '{postal_code}'")
        if mincode:
            filters.append(f"mincode eq '{mincode}'")
        
        filter_expression = " and ".join(filters) if filters else None
        
        # 4. Create vector query
        vector_query = VectorizedQuery(
            vector=name_embedding,
            k_nearest_neighbors=top * 2,  # Get more for reranking
            fields="nameEmbedding"
        )
        
        try:
            # 5. Execute hybrid search
            results = self.search_client.search(
                search_text=keyword_text,
                vector_queries=[vector_query],
                top=top * 2,  # Get more results for reranking
                filter=filter_expression,
                select=[
                    "id", "student_id", "pen", "legalFirstName", "legalMiddleNames", 
                    "legalLastName", "dob", "sexCode", "postalCode", "mincode", 
                    "grade", "localID", "content"
                ],
                scoring_profile="identity-ranking"
            )
            
            # 6. Process and rerank results
            search_results = []
            for result in results:
                # Calculate individual scores (simplified)
                search_score = result.get('@search.score', 0)
                
                # Create SearchResult object
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
                    content=result.get('content'),
                    search_score=search_score
                )
                
                search_results.append(search_result)
            
            # 7. Apply final ranking and return top results
            return search_results[:top]
            
        except Exception as e:
            print(f"Search error: {e}")
            return []

    async def vector_only_search(self, 
                                first_name: str, 
                                last_name: str, 
                                middle_names: str = "",
                                top: int = 10) -> List[SearchResult]:
        """Pure vector search for name similarity"""
        
        name_embedding = self.generate_name_embedding(first_name, last_name, middle_names)
        
        vector_query = VectorizedQuery(
            vector=name_embedding,
            k_nearest_neighbors=top,
            fields="nameEmbedding"
        )
        
        try:
            results = self.search_client.search(
                search_text="*",
                vector_queries=[vector_query],
                top=top,
                select=[
                    "id", "student_id", "pen", "legalFirstName", "legalMiddleNames", 
                    "legalLastName", "dob", "sexCode", "postalCode", "mincode", 
                    "grade", "localID", "content"
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
                    content=result.get('content'),
                    search_score=result.get('@search.score', 0)
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    async def keyword_only_search(self, 
                                 query_text: str,
                                 top: int = 10) -> List[SearchResult]:
        """Pure keyword search fallback"""
        
        try:
            results = self.search_client.search(
                search_text=query_text,
                top=top,
                select=[
                    "id", "student_id", "pen", "legalFirstName", "legalMiddleNames", 
                    "legalLastName", "dob", "sexCode", "postalCode", "mincode", 
                    "grade", "localID", "content"
                ],
                scoring_profile="identity-ranking"
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
                    content=result.get('content'),
                    search_score=result.get('@search.score', 0)
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []

    async def filter_search(self, 
                           filters: Dict[str, str],
                           top: int = 10) -> List[SearchResult]:
        """Search with metadata filters only"""
        
        filter_expressions = []
        for field, value in filters.items():
            filter_expressions.append(f"{field} eq '{value}'")
        
        filter_expression = " and ".join(filter_expressions) if filter_expressions else None
        
        try:
            results = self.search_client.search(
                search_text="*",
                filter=filter_expression,
                top=top,
                select=[
                    "id", "student_id", "pen", "legalFirstName", "legalMiddleNames", 
                    "legalLastName", "dob", "sexCode", "postalCode", "mincode", 
                    "grade", "localID", "content"
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
                    content=result.get('content'),
                    search_score=result.get('@search.score', 0)
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            print(f"Filter search error: {e}")
            return []

    async def get_by_id(self, student_id: str) -> Optional[SearchResult]:
        """Get specific student by ID"""
        try:
            result = self.search_client.get_document(student_id)
            
            return SearchResult(
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
                content=result.get('content'),
                search_score=1.0
            )
            
        except Exception as e:
            print(f"Get by ID error: {e}")
            return None

    def print_results(self, results: List[SearchResult], title: str = "Search Results"):
        """Pretty print search results"""
        print(f"\n=== {title} ===")
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result.search_score:.4f}):")
            print(f"  Student ID: {result.student_id}")
            print(f"  PEN: {result.pen}")
            print(f"  Name: {result.legal_first_name} {result.legal_middle_names or ''} {result.legal_last_name}".strip())
            print(f"  DOB: {result.dob}")
            print(f"  Sex: {result.sex_code}")
            print(f"  Postal: {result.postal_code}")
            print(f"  Mincode: {result.mincode}")
            print("  " + "-" * 50)


# Test functions
async def test_searches():
    """Test different search strategies"""
    search_service = AzureSearchQuery()
    
    # Test 1: Hybrid search
    print("\n🔍 Testing Hybrid Search (ROBYN ANDERSON)")
    results = await search_service.hybrid_search("ROBYN", "ANDERSON", top=5)
    search_service.print_results(results, "Hybrid Search - ROBYN ANDERSON")
    
    # Test 2: Vector only search
    print("\n🔍 Testing Vector-Only Search (MICHAEL LEE)")
    results = await search_service.vector_only_search("MICHAEL", "LEE", top=5)
    search_service.print_results(results, "Vector Search - MICHAEL LEE")
    
    # Test 3: Keyword only search
    print("\n🔍 Testing Keyword-Only Search")
    results = await search_service.keyword_only_search("JENNIFER LEE", top=5)
    search_service.print_results(results, "Keyword Search - JENNIFER LEE")
    
    # Test 4: Filter search
    print("\n🔍 Testing Filter Search (Sex: F)")
    results = await search_service.filter_search({"sexCode": "F"}, top=5)
    search_service.print_results(results, "Filter Search - Females")
    
    # Test 5: Hybrid with metadata
    print("\n🔍 Testing Hybrid + Metadata (DAVID LEE + Sex: M)")
    results = await search_service.hybrid_search("DAVID", "LEE", sex_code="M", top=5)
    search_service.print_results(results, "Hybrid + Metadata - DAVID LEE (Male)")


if __name__ == "__main__":
    asyncio.run(test_searches())