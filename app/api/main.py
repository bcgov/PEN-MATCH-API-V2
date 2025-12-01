from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
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

class MatchingRecord(BaseModel):
    matchingPEN: str
    studentID: str

class PenMatchResponse(BaseModel):
    matchingRecords: List[MatchingRecord]
    penStatus: str
    penStatusMessage: Optional[str] = None

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

# New PEN match endpoint with specified format
@app.post("/pen-match", response_model=PenMatchResponse)
async def pen_match(student_query: StudentQuery):
    """
    Match students and return PEN numbers in the specified format
    
    Args:
        student_query: Student information for matching
    
    Returns:
        PenMatchResponse: Formatted response with matching PEN records
    """
    try:
        logger.info(f"Received PEN match request for: {student_query.legalFirstName} {student_query.legalLastName}")
        
        # Convert Pydantic model to dict and search
        query_dict = student_query.dict(exclude_unset=True)
        result = search_student_by_query(query_dict)
        
        # Extract matching records from the search results
        matching_records = []
        pen_status = "NM"  # Default to No Match
        pen_status_message = None
        
        if result.get('status') == 'success' and result.get('results'):
            # Transform results to matching records format
            for student in result['results']:
                if 'pen' in student and 'studentID' in student:
                    matching_records.append(MatchingRecord(
                        matchingPEN=str(student['pen']),
                        studentID=str(student['studentID'])
                    ))
            
            # Determine PEN status based on results
            if len(matching_records) == 1:
                pen_status = "EM"  # Exact Match
            elif len(matching_records) > 1:
                pen_status = "CM"  # Confident Match (multiple matches)
            else:
                pen_status = "NM"  # No Match
                pen_status_message = "No matching records found"
        else:
            pen_status_message = result.get('message', 'Search failed')
        
        logger.info(f"PEN match completed - Status: {pen_status}, Matches: {len(matching_records)}")
        
        return PenMatchResponse(
            matchingRecords=matching_records,
            penStatus=pen_status,
            penStatusMessage=pen_status_message
        )
    
    except Exception as e:
        logger.error(f"PEN match error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PEN match failed: {str(e)}"
        )

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)