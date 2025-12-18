from typing_extensions import TypedDict
from typing import Any, Dict, List, Optional

class PenMatchState(TypedDict, total=False):
    # Input request data
    request: Dict[str, Any]
    
    # Candidates from Azure search
    candidates: List[Dict[str, Any]]
    
    # Search metadata (status, type, count, etc.)
    search_metadata: Optional[Dict[str, Any]]
    
    # LLM analysis results
    analysis: Optional[Dict[str, Any]]
    
    # Final decision outputs
    final_decision: Optional[str]  # CONFIRM, REVIEW, NO_MATCH
    selected_candidate: Optional[str]  # student_id if chosen
    confidence: Optional[float]
    
    # Routing and processing info
    route: Optional[str]  # routing decision
    llm_used: Optional[bool]  # whether LLM was used
    
    # Error handling
    error: Optional[str]
