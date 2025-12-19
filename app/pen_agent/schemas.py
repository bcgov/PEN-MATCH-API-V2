# pen_agent/schemas.py
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

Decision = Literal["CONFIRM", "REVIEW", "NO_MATCH"]
Severity = Literal["high", "medium", "low"]
IssueType = Literal["typo", "outdated", "missing", "conflict", "formatting"]


class KeyValue(BaseModel):
    """Strict-safe replacement for Dict[str, Any]."""
    model_config = ConfigDict(extra="forbid")
    key: str
    value: Optional[str] = None


class Mismatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    detail: str
    severity: Severity = "medium"


class SuspectedInputIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    issue: IssueType
    hint: str


class CandidateRecord(BaseModel):
    """
    Full candidate reference:
    - core fields are explicit
    - extras are a small allowlist (stringified)
    """
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(..., description="1-based rank in candidates_json list")

    student_id: str
    pen: Optional[str] = None
    legalFirstName: Optional[str] = None
    legalMiddleNames: Optional[str] = None
    legalLastName: Optional[str] = None
    dob: Optional[str] = None
    sexCode: Optional[str] = None
    mincode: Optional[str] = None
    postalCode: Optional[str] = None
    localID: Optional[str] = None
    gradeCode: Optional[str] = None

    search_score: Optional[float] = None
    final_score: Optional[float] = None
    search_method: Optional[str] = None

    extras: List[KeyValue] = Field(default_factory=list)


class ReviewCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate: CandidateRecord
    reasons: List[str] = Field(default_factory=list)
    issues: List[Mismatch] = Field(default_factory=list)


class CandidateAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    confidence: float = Field(..., ge=0.0, le=1.0)

    reasons: List[str] = Field(default_factory=list)

    chosen_candidate: Optional[CandidateRecord] = None

    # global blockers summary
    mismatches: List[Mismatch] = Field(default_factory=list)

    # REVIEW must return up to 5
    review_candidates: List[ReviewCandidate] = Field(default_factory=list)

    suspected_input_issues: List[SuspectedInputIssue] = Field(default_factory=list)
