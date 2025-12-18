from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AliasChoices
from pydantic import ConfigDict
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
from pathlib import Path

from azure_search.azure_search_query import search_student_by_query, StudentSearchService

# -------------------------------------------------------------------
# Logging setup (auto-create logs directory)
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # /app
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "student_match.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="PEN Match API V2",
    description="API for matching students and retrieving PEN numbers",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def normalize_dob(dob: Optional[str]) -> Optional[str]:
    """
    Accept both '20010210' and '2001-02-10' and normalize to 'YYYY-MM-DD'.
    If format is unrecognized, return as-is.
    """
    if not dob:
        return None

    dob = dob.strip()
    try:
        # Old format: YYYYMMDD
        if len(dob) == 8 and dob.isdigit():
            dt = datetime.strptime(dob, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        # New format: YYYY-MM-DD
        if len(dob) == 10 and dob[4] == "-" and dob[7] == "-":
            dt = datetime.strptime(dob, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning(f"Failed to parse dob='{dob}': {e}")

    # Fallback: leave unchanged if we can't parse
    return dob


def build_query_dict(student_query: "StudentQuery") -> Dict[str, Any]:
    """
    Convert StudentQuery to a clean dict:
    - use internal field names (legalFirstName, legalLastName, dob, etc.)
    - drop None / "" / [] values
    - normalize dob format
    """
    raw = student_query.model_dump(exclude_unset=True, by_alias=False)
    cleaned: Dict[str, Any] = {}

    for k, v in raw.items():
        if v in (None, "", []):
            continue
        if k == "dob":
            v = normalize_dob(v)
            if not v:
                continue
        cleaned[k] = v

    return cleaned


def get_pen_status_message(pen_status: str) -> Optional[str]:
    """
    Convert pen_status codes to human-readable messages.
    Returns None for certain status codes to match expected response format.
    """
    status_messages = {
        "AA": None,  # "All provided fields match exactly",
        "BM": None,  # "Multiple fields match but some differences found",
        "F1": None,  # "Only one or zero fields match",
        "D1": None,  # Single match - no message needed
        "CM": None,  # Multiple matches - no message needed  
        "C0": None,  # "Too many candidates or no matches found"
    }
    return status_messages.get(pen_status, "Unknown status")


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------
class StudentQuery(BaseModel):
    """
    Backward-compatible model:

    Required (must be present as either new or legacy names):
      - legalFirstName  ← legalFirstName OR givenName
      - legalLastName   ← legalLastName OR surname

    Optional fields can come from either new or legacy keys:
      - legalMiddleNames  ← legalMiddleNames OR middleName
      - sexCode           ← sexCode OR sex
      - postalCode        ← postalCode OR postal
      - gradeCode         ← gradeCode OR enrolledGradeCode
      - dob               ← dob (either 'YYYYMMDD' or 'YYYY-MM-DD')
      - mincode, localID, pen as-is

    Unknown/extra fields (usualSurname, assignNewPEN, etc.) are ignored.
    """

    model_config = ConfigDict(extra="ignore")  # ignore unknown fields

    # Required
    legalFirstName: str = Field(
        ...,
        description="Legal first name of the student",
        validation_alias=AliasChoices("legalFirstName", "givenName"),
    )
    legalLastName: str = Field(
        ...,
        description="Legal last name of the student",
        validation_alias=AliasChoices("legalLastName", "surname"),
    )

    # Optional
    legalMiddleNames: Optional[str] = Field(
        None,
        description="Legal middle names of the student",
        validation_alias=AliasChoices("legalMiddleNames", "middleName"),
    )
    dob: Optional[str] = Field(
        None,
        description="Date of birth, 'YYYY-MM-DD' or 'YYYYMMDD'",
        validation_alias=AliasChoices("dob", "DOB"),
    )
    localID: Optional[str] = Field(
        None,
        description="Local student ID",
        validation_alias=AliasChoices("localID", "localID"),
    )
    sexCode: Optional[str] = Field(
        None,
        description="Sex code",
        validation_alias=AliasChoices("sexCode", "sex"),
    )
    postalCode: Optional[str] = Field(
        None,
        description="Postal code",
        validation_alias=AliasChoices("postalCode", "postal"),
    )
    mincode: Optional[str] = Field(
        None,
        description="Mincode",
        validation_alias=AliasChoices("mincode", "mincode"),
    )
    pen: Optional[str] = Field(
        None,
        description="PEN number",
        validation_alias=AliasChoices("pen", "pen"),
    )
    gradeCode: Optional[str] = Field(
        None,
        description="Grade code",
        validation_alias=AliasChoices("gradeCode", "enrolledGradeCode"),
    )


class SearchQuery(BaseModel):
    """
    Optional search model (not currently used in endpoints, but kept for future).
    Also backward-compatible with legacy names.
    """

    model_config = ConfigDict(extra="ignore")

    legalFirstName: Optional[str] = Field(
        None, validation_alias=AliasChoices("legalFirstName", "givenName")
    )
    legalMiddleNames: Optional[str] = Field(
        None, validation_alias=AliasChoices("legalMiddleNames", "middleName")
    )
    legalLastName: Optional[str] = Field(
        None, validation_alias=AliasChoices("legalLastName", "surname")
    )
    dob: Optional[str] = Field(
        None, validation_alias=AliasChoices("dob", "DOB")
    )
    localID: Optional[str] = Field(
        None, validation_alias=AliasChoices("localID", "localID")
    )
    sexCode: Optional[str] = Field(
        None, validation_alias=AliasChoices("sexCode", "sex")
    )
    postalCode: Optional[str] = Field(
        None, validation_alias=AliasChoices("postalCode", "postal")
    )
    mincode: Optional[str] = Field(
        None, validation_alias=AliasChoices("mincode", "mincode")
    )
    pen: Optional[str] = Field(
        None, validation_alias=AliasChoices("pen", "pen")
    )
    gradeCode: Optional[str] = Field(
        None, validation_alias=AliasChoices("gradeCode", "enrolledGradeCode")
    )


class MatchingRecord(BaseModel):
    matchingPEN: str
    studentID: str


class PenMatchResponse(BaseModel):
    matchingRecords: List[MatchingRecord]
    penStatus: str
    penStatusMessage: Optional[str] = None


class MatchResponse(BaseModel):
    status: str
    pen_status: Optional[str] = None
    search_type: Optional[str] = None
    count: Optional[int] = None
    message: Optional[str] = None
    results: Optional[list] = None
    methodology: Optional[dict] = None


# -------------------------------------------------------------------
# Root / health endpoints
# -------------------------------------------------------------------
@app.get("/")
async def root():
    """Health check / root endpoint."""
    return {
        "message": "PEN Match API V2 is running",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "PEN Match API V2",
    }


# -------------------------------------------------------------------
# PEN match endpoint (formatted PEN response)
# -------------------------------------------------------------------
@app.post("/pen-match", response_model=PenMatchResponse)
async def pen_match(student_query: StudentQuery):
    """
    Match students and return PEN numbers in the specified format.

    Only legalFirstName / givenName and legalLastName / surname are mandatory.
    All other fields are optional and may use legacy names.

    PEN Status Codes:
    - AA: All provided fields match exactly
    - BM: Multiple fields match but some differences found  
    - F1: Only one or zero fields match
    - D1: Single exact match found
    - CM: Multiple matches found
    - C0: Too many candidates or no matches found
    """
    try:
        logger.info(
            f"Received PEN match request for: "
            f"{student_query.legalFirstName} {student_query.legalLastName}"
        )

        query_dict = build_query_dict(student_query)
        result = search_student_by_query(query_dict)

        # Extract pen_status directly from search result (new implementation)
        pen_status = result.get("pen_status", "C0")  # Default to C0 if not found
        
        matching_records: List[MatchingRecord] = []

        # Process results if available
        if result.get("status") == "success" and result.get("results"):
            for student in result["results"]:
                pen_val = student.get("pen")
                # Support different ID field names from Azure Search
                student_id_val = (
                    student.get("studentID")
                    or student.get("student_id")
                    or student.get("id")
                    or student.get("studentId")  # alternative casing
                )

                if pen_val and student_id_val:
                    matching_records.append(
                        MatchingRecord(
                            matchingPEN=str(pen_val),
                            studentID=str(student_id_val),
                        )
                    )

        # Get status message based on pen_status
        pen_status_message = get_pen_status_message(pen_status)
        
        # Override message for C0 when no results found
        if pen_status == "C0" and not matching_records and result.get("message"):
            pen_status_message = result.get("message")

        logger.info(
            f"PEN match completed - PEN Status: {pen_status}, "
            f"Matches: {len(matching_records)}"
        )

        return PenMatchResponse(
            matchingRecords=matching_records,
            penStatus=pen_status,
            penStatusMessage=pen_status_message,
        )

    except Exception as e:
        logger.error(f"PEN match error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PEN match failed: {str(e)}",
        )


# -------------------------------------------------------------------
# Original student matching endpoint (raw search response)
# -------------------------------------------------------------------
@app.post("/match-student", response_model=MatchResponse)
async def match_student(student_query: StudentQuery):
    """
    Match a student using Azure Search (original endpoint).

    Returns the raw search response structure with new pen_status codes:
    - AA: All provided fields match exactly
    - BM: Multiple fields match but some differences found  
    - F1: Only one or zero fields match
    - D1: Single exact match found
    - CM: Multiple matches found
    - C0: Too many candidates or no matches found
    """
    try:
        logger.info(
            f"Received match request for: "
            f"{student_query.legalFirstName} {student_query.legalLastName}"
        )

        query_dict = build_query_dict(student_query)
        result = search_student_by_query(query_dict)

        logger.info(
            f"Search completed - Status: {result.get('status')}, "
            f"PEN Status: {result.get('pen_status')}, "
            f"Type: {result.get('search_type')}, Count: {result.get('count')}"
        )

        return MatchResponse(**result)

    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)