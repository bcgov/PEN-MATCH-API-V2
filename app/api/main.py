from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from azure_search.azure_search_query import search_student_by_query, StudentSearchService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app/logs/student_match.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PEN Match API V2",
    description="API for matching students and retrieving PEN numbers",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class StudentQuery(BaseModel):
    legalFirstName: str = Field(..., description="Legal first name of the student")
    legalMiddleNames: Optional[str] = Field("", description="Legal middle names of the student")
    legalLastName: str = Field(..., description="Legal last name of the student")
    dob: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    localID: str = Field(..., description="Local student ID")
    sexCode: Optional[str] = Field(None, description="Sex code")
    postalCode: Optional[str] = Field(None, description="Postal code")
    mincode: Optional[str] = Field(None, description="Mincode")
    pen: Optional[str] = Field(None, description="PEN number")

class SearchQuery(BaseModel):
    legalFirstName: Optional[str] = Field(None, description="Legal first name")
    legalMiddleNames: Optional[str] = Field(None, description="Legal middle names")
    legalLastName: Optional[str] = Field(None, description="Legal last name")
    dob: Optional[str] = Field(None, description="Date of birth (YYYY-MM-DD)")
    localID: Optional[str] = Field(None, description="Local student ID")
    sexCode: Optional[str] = Field(None, description="Sex code")
    postalCode: Optional[str] = Field(None, description="Postal code")
    mincode: Optional[str] = Field(None, description="Mincode")
    pen: Optional[str] = Field(None, description="PEN number")
    gradeCode: Optional[str] = Field(None, description="Grade code")

class MatchResponse(BaseModel):
    status: str
    search_type: Optional[str] = None
    count: Optional[int] = None
    message: Optional[str] = None
    results: Optional[list] = None
    methodology: Optional[dict] = None

# Root endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "PEN Match API V2 is running", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "PEN Match API V2"
    }

# Original student matching endpoint
@app.post("/match-student", response_model=MatchResponse)
async def match_student(student_query: StudentQuery):
    """
    Match a student using Azure Search (original endpoint)
    
    Args:
        student_query: Student information for matching
    
    Returns:
        MatchResponse: Direct response from Azure Search
    """
    try:
        logger.info(f"Received match request for: {student_query.legalFirstName} {student_query.legalLastName}")
        
        # Convert Pydantic model to dict and search
        query_dict = student_query.dict(exclude_unset=True)
        result = search_student_by_query(query_dict)
        
        logger.info(f"Search completed - Status: {result.get('status')}, Type: {result.get('search_type')}, Count: {result.get('count')}")
        
        return MatchResponse(**result)
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

# Azure Search specific endpoints
@app.post("/azure-search/search", response_model=MatchResponse, tags=["Azure Search"])
async def search_students(query: SearchQuery):
    """
    Search students using Azure Search with exact and fuzzy matching
    
    Supports both exact matching (hard filter) and fuzzy matching (vector-based)
    """
    try:
        query_dict = query.dict(exclude_unset=True)
        result = search_student_by_query(query_dict)
        
        logger.info(f"Azure Search completed - Type: {result.get('search_type')}, Count: {result.get('count')}")
        
        return MatchResponse(**result)
    
    except Exception as e:
        logger.error(f"Azure Search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.get("/azure-search/search", response_model=MatchResponse, tags=["Azure Search"])
async def search_students_get(
    pen: Optional[str] = Query(None, description="PEN number"),
    legal_first_name: Optional[str] = Query(None, alias="legalFirstName"),
    legal_middle_names: Optional[str] = Query(None, alias="legalMiddleNames"),
    legal_last_name: Optional[str] = Query(None, alias="legalLastName"),
    dob: Optional[str] = Query(None, description="Date of birth (YYYY-MM-DD)"),
    local_id: Optional[str] = Query(None, alias="localID"),
    sex_code: Optional[str] = Query(None, alias="sexCode"),
    postal_code: Optional[str] = Query(None, alias="postalCode"),
    mincode: Optional[str] = Query(None),
    grade_code: Optional[str] = Query(None, alias="gradeCode"),
):
    """
    Search students via GET parameters
    """
    try:
        query_dict = {
            k: v for k, v in {
                "pen": pen,
                "legalFirstName": legal_first_name,
                "legalMiddleNames": legal_middle_names,
                "legalLastName": legal_last_name,
                "dob": dob,
                "localID": local_id,
                "sexCode": sex_code,
                "postalCode": postal_code,
                "mincode": mincode,
                "gradeCode": grade_code,
            }.items() if v is not None
        }
        
        if not query_dict:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one search parameter must be provided"
            )
        
        result = search_student_by_query(query_dict)
        return MatchResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Azure Search GET error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.get("/azure-search/health", tags=["Azure Search"])
async def azure_search_health():
    """Check Azure Search service health"""
    try:
        service = StudentSearchService()
        return {
            "status": "healthy",
            "service": "Azure Search",
            "endpoint": service.search_endpoint,
            "index": service.index_name,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Azure Search health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Azure Search service unavailable: {str(e)}"
        )

# Additional utility endpoints
@app.get("/azure-search/test", tags=["Azure Search"])
async def test_azure_search():
    """Test Azure Search with a known query"""
    try:
        # Test with a simple query
        test_query = {"legalFirstName": "MICHAEL", "legalLastName": "LEE"}
        result = search_student_by_query(test_query)
        
        return {
            "test_status": "completed",
            "search_type": result.get('search_type'),
            "count": result.get('count'),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Azure Search test failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)