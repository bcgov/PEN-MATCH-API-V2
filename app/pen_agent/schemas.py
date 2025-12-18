from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

Decision = Literal["CONFIRM", "REVIEW", "NO_MATCH"]

class CandidateAnalysis(BaseModel):
    decision: Decision
    chosen_student_id: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = []
    mismatches: Dict[str, Any] = {}
