from typing import Dict, Any
from azure_search.azure_search_query import search_student_by_query
from .schemas import CandidateAnalysis
from .prompt import ANALYZE_CANDIDATES_PROMPT
from .llm_client import LLMClient

def fetch_candidates_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch candidates using Azure Search based on the request data
    """
    request = state["request"]
    
    try:
        # Use your Azure search service to get candidates
        search_result = search_student_by_query(request)
        
        # Extract candidates from search result
        candidates = search_result.get("results", [])
        
        print(f"DEBUG: Found {len(candidates)} candidates from Azure Search")
        
        # Debug print all candidate information (excluding embeddings)
        for i, candidate in enumerate(candidates, 1):
            print(f"\nDEBUG: Candidate {i} details:")
            # Filter out embedding fields for debug output
            debug_candidate = {k: v for k, v in candidate.items() 
                             if not k.lower().endswith('embedding') and not k.lower().endswith('vector')}
            print(f"  Student ID: {candidate.get('student_id')}")
            print(f"  PEN: {candidate.get('pen')}")
            print(f"  Name: {candidate.get('legalFirstName')} {candidate.get('legalMiddleNames', '')} {candidate.get('legalLastName')}")
            print(f"  DOB: {candidate.get('dob')}")
            print(f"  Sex: {candidate.get('sexCode')}")
            print(f"  Postal Code: {candidate.get('postalCode')}")
            print(f"  Mincode: {candidate.get('mincode')}")
            print(f"  Local ID: {candidate.get('localID')}")
            print(f"  Grade: {candidate.get('gradeCode')}")
            print(f"  Full data: {debug_candidate}")
        
        return {
            "candidates": candidates,
            "search_metadata": {
                "status": search_result.get("status"),
                "search_type": search_result.get("search_type"),
                "count": search_result.get("count", 0),
                "pen_status": search_result.get("pen_status")
            }
        }
    except Exception as e:
        print(f"Error fetching candidates: {e}")
        return {
            "candidates": [],
            "search_metadata": {
                "status": "error",
                "error": str(e)
            }
        }

def decision_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Router to decide if LLM analysis is needed based on candidate count
    """
    candidates = state["candidates"]
    candidate_count = len(candidates)
    
    print(f"DEBUG: Decision router - found {candidate_count} candidates")
    
    if candidate_count == 0:
        print("DEBUG: No candidates found - returning NO_MATCH")
        return {
            "final_decision": "NO_MATCH",
            "selected_candidate": None,
            "confidence": 0.0,
            "llm_used": False,
            "analysis": {
                "decision": "NO_MATCH",
                "confidence": 0.0,
                "reasons": ["No candidates found in search"],
                "mismatches": {}
            },
            "route": "end"
        }
    elif candidate_count == 1:
        print("DEBUG: Single candidate found - auto-confirming without LLM")
        candidate = candidates[0]
        return {
            "final_decision": "CONFIRM",
            "selected_candidate": candidate.get('student_id'),
            "confidence": 1.0,
            "llm_used": False,
            "analysis": {
                "decision": "CONFIRM",
                "confidence": 1.0,
                "reasons": ["Single exact match found from search"],
                "chosen_student_id": candidate.get('student_id'),
                "mismatches": {}
            },
            "route": "end"
        }
    else:
        print(f"DEBUG: Multiple candidates ({candidate_count}) found - routing to LLM analysis")
        return {
            "route": "llm_analyze",
            "llm_used": True
        }

def llm_analyze_node(state: Dict[str, Any], llm_client: LLMClient) -> Dict[str, Any]:
    """
    Analyze candidates using LLM to make final decision
    Only called when multiple candidates are found
    """
    request = state["request"]
    candidates = state["candidates"]
    
    print(f"DEBUG: LLM Analysis starting with {len(candidates)} candidates")
    
    # Get all original data (excluding embeddings) for LLM analysis
    processed_candidates = []
    for candidate in candidates:
        # Remove embedding/vector fields for LLM analysis
        clean_candidate = {
            k: v for k, v in candidate.items() 
            if not k.lower().endswith('embedding') and not k.lower().endswith('vector')
        }
        processed_candidates.append(clean_candidate)
    
    print(f"DEBUG: Prepared {len(processed_candidates)} candidates for LLM analysis")
    
    # Format the prompt
    prompt = ANALYZE_CANDIDATES_PROMPT.format(
        request=request,
        candidates=processed_candidates
    )
    
    print(f"DEBUG: Sending prompt to LLM (length: {len(prompt)} chars)")
    
    try:
        # Use structured output to ensure proper response format
        result = llm_client.with_structured_output(CandidateAnalysis).invoke(prompt)
        
        print(f"DEBUG: LLM returned decision: {result.decision} with confidence: {result.confidence}")
        
        return {
            "analysis": result.model_dump(),
            "final_decision": result.decision,
            "selected_candidate": result.chosen_student_id,
            "confidence": result.confidence,
            "llm_used": True,
            "route": "end"
        }
    except Exception as e:
        print(f"Error in LLM analysis: {e}")
        return {
            "analysis": {
                "decision": "NO_MATCH",
                "confidence": 0.0,
                "reasons": [f"LLM analysis failed: {str(e)}"],
                "mismatches": {}
            },
            "final_decision": "NO_MATCH",
            "selected_candidate": None,
            "confidence": 0.0,
            "llm_used": True,
            "error": str(e),
            "route": "end"
        }
