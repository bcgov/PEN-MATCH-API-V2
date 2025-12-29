from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AliasChoices, ConfigDict
from typing import Optional, Dict, Any, List
import logging
import json
from datetime import datetime
from pathlib import Path

from pen_agent.workflow import create_pen_match_workflow

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pen_agent_api.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="PEN Match Agent API",
    description="Intelligent PEN matching API with comprehensive LLM analysis",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class StudentQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    legalFirstName: str = Field(..., validation_alias=AliasChoices("legalFirstName", "givenName"))
    legalLastName: str = Field(..., validation_alias=AliasChoices("legalLastName", "surname"))
    legalMiddleNames: Optional[str] = Field(None, validation_alias=AliasChoices("legalMiddleNames", "middleName"))
    dob: Optional[str] = Field(None, validation_alias=AliasChoices("dob", "DOB"))
    localID: Optional[str] = Field(None, validation_alias=AliasChoices("localID", "localID"))
    sexCode: Optional[str] = Field(None, validation_alias=AliasChoices("sexCode", "sex"))
    postalCode: Optional[str] = Field(None, validation_alias=AliasChoices("postalCode", "postal"))
    gradeCode: Optional[str] = Field(None, validation_alias=AliasChoices("gradeCode", "enrolledGradeCode"))
    mincode: Optional[str] = Field(None, validation_alias=AliasChoices("mincode", "mincode"))
    pen: Optional[str] = Field(None, validation_alias=AliasChoices("pen", "pen"))


class CandidateInfo(BaseModel):
    rank: int
    student_id: str
    pen: Optional[str] = None
    name: str
    dob: Optional[str] = None
    search_score: Optional[float] = None
    final_score: Optional[float] = None


class AnalysisResponse(BaseModel):
    success: bool
    timestamp: str
    request: Dict[str, Any]
    final_decision: str
    confidence: float
    llm_used: bool
    model_used: str
    candidates_found: int
    
    # Analysis details
    reasoning: List[str] = Field(default_factory=list)
    chosen_candidate: Optional[CandidateInfo] = None
    review_candidates: List[CandidateInfo] = Field(default_factory=list)
    mismatches: List[Dict[str, Any]] = Field(default_factory=list)
    input_issues: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    error: Optional[str] = None


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def convert_query_to_legacy_format(student_query: StudentQuery) -> Dict[str, Any]:
    return {
        "givenName": student_query.legalFirstName,
        "surname": student_query.legalLastName,
        "middleName": student_query.legalMiddleNames or "",
        "dob": student_query.dob or "",
        "localID": student_query.localID or "",
        "sex": student_query.sexCode or "",
        "postal": student_query.postalCode or "",
        "enrolledGradeCode": student_query.gradeCode or "",
        "mincode": student_query.mincode or "",
        "pen": student_query.pen or "",
    }


def format_candidate_name(candidate: Dict[str, Any]) -> str:
    first = candidate.get("legalFirstName", "")
    middle = candidate.get("legalMiddleNames", "")
    last = candidate.get("legalLastName", "")
    return f"{first} {middle} {last}".strip() if middle else f"{first} {last}".strip()


def convert_candidate(candidate: Dict[str, Any]) -> CandidateInfo:
    return CandidateInfo(
        rank=candidate.get("rank", 0),
        student_id=candidate.get("student_id", ""),
        pen=candidate.get("pen"),
        name=format_candidate_name(candidate),
        dob=candidate.get("dob"),
        search_score=candidate.get("search_score"),
        final_score=candidate.get("final_score"),
    )


def generate_recommendations(analysis: Dict[str, Any], decision: str, confidence: float) -> List[str]:
    recommendations = []
    
    if decision == "CONFIRM":
        if confidence > 0.9:
            recommendations.append("High confidence match - proceed with PEN assignment")
        else:
            recommendations.append("Match found - verify details before proceeding")
    elif decision == "REVIEW":
        recommendations.append("Multiple potential matches - manual review required")
        recommendations.append("Compare candidates using provided analysis")
    else:
        recommendations.append("No suitable matches - verify input data accuracy")
        if analysis.get("suspected_input_issues"):
            recommendations.append("Check suspected input issues below")
            
    return recommendations


# -------------------------------------------------------------------
# API endpoint
# -------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PEN Match Agent API is running",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "endpoints": ["/analyze", "/docs"]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "PEN Match Agent API"
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_match(student_query: StudentQuery):
    """
    Comprehensive PEN matching analysis with LLM-powered intelligence.
    Provides detailed candidate analysis, decision rationale, and recommendations.
    """
    timestamp = datetime.now().isoformat()
    
    try:
        logger.info(f"Full analysis for: {student_query.legalFirstName} {student_query.legalLastName}")
        
        request_data = convert_query_to_legacy_format(student_query)
        workflow = create_pen_match_workflow()
        result = workflow.run_match(request_data)
        
        if not result.get("success", True):
            error_msg = result.get("error", "Unknown workflow error")
            return AnalysisResponse(
                success=False,
                timestamp=timestamp,
                request=request_data,
                final_decision="NO_MATCH",
                confidence=0.0,
                llm_used=False,
                model_used="none",
                candidates_found=0,
                error=error_msg,
                recommendations=["Check system configuration and retry"]
            )
        
        analysis_data = result.get("analysis", {})
        final_decision = result.get("final_decision", "NO_MATCH")
        confidence = result.get("confidence", 0.0)
        
        # Convert candidates
        chosen_candidate = None
        review_candidates = []
        
        if analysis_data.get("chosen_candidate"):
            chosen_candidate = convert_candidate(analysis_data["chosen_candidate"])
            
        for rc in analysis_data.get("review_candidates", []):
            review_candidates.append(convert_candidate(rc.get("candidate", {})))
        
        recommendations = generate_recommendations(analysis_data, final_decision, confidence)
        
        logger.info(f"Analysis complete - Decision: {final_decision}, Confidence: {confidence:.2f}")
        
        return AnalysisResponse(
            success=True,
            timestamp=timestamp,
            request=request_data,
            final_decision=final_decision,
            confidence=confidence,
            llm_used=result.get("llm_used", False),
            model_used=result.get("model_used", "unknown"),
            candidates_found=result.get("candidates_count", 0),
            reasoning=analysis_data.get("reasons", []),
            chosen_candidate=chosen_candidate,
            review_candidates=review_candidates,
            mismatches=analysis_data.get("mismatches", []),
            input_issues=analysis_data.get("suspected_input_issues", []),
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        
        return AnalysisResponse(
            success=False,
            timestamp=timestamp,
            request=convert_query_to_legacy_format(student_query),
            final_decision="NO_MATCH",
            confidence=0.0,
            llm_used=False,
            model_used="none",
            candidates_found=0,
            error=str(e),
            recommendations=["Check input data and retry"]
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)