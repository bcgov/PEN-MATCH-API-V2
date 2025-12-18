# pen_agent/schemas.py
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

Decision = Literal["CONFIRM", "REVIEW", "NO_MATCH"]
Severity = Literal["high", "medium", "low"]
Plausibility = Literal["plausible", "unlikely"]


class Mismatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(..., description="Field name that conflicts or is uncertain.")
    detail: str = Field(..., description="Short explanation of the issue.")
    severity: Severity = "medium"


class SuspectedInputIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str = Field(..., description="Which request field likely has an issue.")
    issue: Literal["typo", "outdated", "missing", "conflict", "formatting"]
    hint: str


class CandidateRecord(BaseModel):
    """
    Candidate record that the LLM is allowed to echo back.
    Keep this aligned with what you pass into candidates_json.
    """
    model_config = ConfigDict(extra="forbid")

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

    # Optional debug scores (if you pass them)
    search_score: Optional[float] = None
    final_score: Optional[float] = None
    search_method: Optional[str] = None


class RankedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: CandidateRecord
    plausibility: Plausibility
    summary: str = Field(..., description="One-line summary why plausible/unlikely.")
    issues: List[Mismatch] = Field(default_factory=list)


class CandidateAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)

    # If CONFIRM, return the full chosen record including pen
    chosen_candidate: Optional[CandidateRecord] = None

    # General mismatches/uncertainties for the best candidate(s)
    mismatches: List[Mismatch] = Field(default_factory=list)

    # If REVIEW, return up to 5 plausible/unlikely candidates with issues
    ranked_candidates: List[RankedCandidate] = Field(default_factory=list)

    # If NO_MATCH or weak REVIEW: which request fields to re-check
    suspected_input_issues: List[SuspectedInputIssue] = Field(default_factory=list)
