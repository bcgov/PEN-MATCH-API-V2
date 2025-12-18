# pen_agent/schemas.py
from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

Decision = Literal["CONFIRM", "REVIEW", "NO_MATCH"]
Severity = Literal["high", "medium", "low"]
IssueType = Literal["typo", "outdated", "missing", "conflict", "formatting"]


class KeyValue(BaseModel):
    """Safe replacement for Dict[str, Any] in strict structured output."""
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
    Candidate record that LLM can echo back.
    Core fields + extras (stringified) to support "all fields information".
    """
    model_config = ConfigDict(extra="forbid")

    # core identifiers
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

    # optional scoring/debug
    search_score: Optional[float] = None
    final_score: Optional[float] = None
    search_method: Optional[str] = None

    # any remaining fields from search, as strings
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

    # CONFIRM: filled; otherwise null
    chosen_candidate: Optional[CandidateRecord] = None

    # top blockers / mistakes (keep this for the structure you like)
    mismatches: List[Mismatch] = Field(default_factory=list)

    # REVIEW: default top 5 plausible candidates (can include "unlikely" if needed, but keep list reasonable)
    review_candidates: List[ReviewCandidate] = Field(default_factory=list)

    # NO_MATCH: suggested fields to re-check
    suspected_input_issues: List[SuspectedInputIssue] = Field(default_factory=list)
