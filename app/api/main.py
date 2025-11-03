from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
from datetime import datetime

from app.core.student_match import StudentWorkflow

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

class MatchResponse(BaseModel):
    status: str
    pen: Optional[str] = None
    similarity_score: Optional[float] = None
    message: Optional[str] = None
    source: Optional[str] = None
    candidates_count: Optional[int] = None

# Initialize workflow
workflow = StudentWorkflow()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "PEN Match API V2 is running", "timestamp": datetime.now().isoformat()}

@app.post("/match-student", response_model=MatchResponse)
async def match_student(student_query: StudentQuery):
    """
    Match a student and return their PEN number if found
    
    Args:
        student_query: Student information for matching
    
    Returns:
        MatchResponse: Contains PEN number if match found, or error details
    """
    try:
        logger.info(f"Received match request for: {student_query.legalFirstName} {student_query.legalLastName}")
        
        # Convert Pydantic model to dict
        query_student = student_query.dict()
        
        # Process the student query
        result = workflow.process_student_query(query_student)
        
        logger.info(f"Match result status: {result['status']}")
        
        # Handle different result types
        if result['status'] == 'perfect_match_found':
            pen = result['student'].get('pen')
            logger.info(f"Perfect match found - PEN: {pen}, Score: {result['similarity_score']:.4f}")
            
            return MatchResponse(
                status="perfect_match_found",
                pen=pen,
                similarity_score=result['similarity_score'],
                message=f"Perfect match found with {result['similarity_score']:.4f} similarity",
                source=result['source']
            )
        
        elif result['status'] == 'no_perfect_match':
            candidates_count = len(result.get('candidates', []))
            logger.warning(f"No perfect match found. Best score: {result['best_score']:.4f}, Candidates: {candidates_count}")
            
            return MatchResponse(
                status="no_perfect_match",
                similarity_score=result['best_score'],
                message=f"No perfect match found. Best similarity: {result['best_score']:.4f}",
                source=result['source'],
                candidates_count=candidates_count
            )
        
        elif result['status'] == 'no_students_found':
            logger.warning(f"No students found: {result['message']}")
            
            return MatchResponse(
                status="no_students_found",
                message=result['message']
            )
        
        else:
            logger.error(f"Unexpected result status: {result['status']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected result status: {result['status']}"
            )
    
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/bulk-import")
async def bulk_import_students(page_size: int = 100, max_pages: Optional[int] = None):
    """
    Bulk import students from source database to Cosmos DB
    
    Args:
        page_size: Number of students per page
        max_pages: Maximum number of pages to process (optional)
    
    Returns:
        Import statistics
    """
    try:
        logger.info(f"Starting bulk import - page_size: {page_size}, max_pages: {max_pages}")
        
        total_imported = workflow.bulk_import_students(page_size=page_size, max_pages=max_pages)
        
        logger.info(f"Bulk import completed - total imported: {total_imported}")
        
        return {
            "status": "completed",
            "total_imported": total_imported,
            "page_size": page_size,
            "max_pages": max_pages
        }
    
    except Exception as e:
        logger.error(f"Bulk import failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk import failed: {str(e)}"
        )





@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    try:
        # Test basic workflow initialization
        test_workflow = StudentWorkflow()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "student_workflow": "operational",
                "cosmos_db": "connected",
                "embedding_service": "operational"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)