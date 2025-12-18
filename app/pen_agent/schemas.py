# pen_agent/schemas.py
from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from typing import List, Optional, Literal

Decision = Literal["CONFIRM", "REVIEW", "NO_MATCH"]


class Mismatch(BaseModel):
    # Important for strict schema: disallow extra keys
    model_config = ConfigDict(extra="forbid")

    field: str = Field(..., description="Which field mismatched or is uncertain.")
    detail: str = Field(..., description="Short explanation of mismatch/uncertainty.")
    severity: Optional[Literal["high", "medium", "low"]] = None


class CandidateAnalysis(BaseModel):
    # Important for strict schema: disallow extra keys
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    chosen_student_id: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)

    reasons: List[str] = Field(default_factory=list)

    # FIX: no Dict. Use a typed list instead.
    mismatches: List[Mismatch] = Field(default_factory=list)
