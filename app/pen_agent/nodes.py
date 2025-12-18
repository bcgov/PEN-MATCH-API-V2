# pen_agent/nodes.py
from typing import Dict, Any, List
import json

from azure_search.azure_search_query import search_student_by_query
from .schemas import CandidateAnalysis
from .prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .llm_client import LLMClient


# Only pass fields that your schema allows (prevents token bloat + validation issues)
ALLOWED_CANDIDATE_FIELDS = {
    "student_id", "pen",
    "legalFirstName", "legalMiddleNames", "legalLastName",
    "dob", "sexCode", "mincode", "postalCode",
    "localID", "gradeCode",
    "@search.score", "final_score", "search_method",
}


def slim_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only safe/relevant fields for LLM analysis, and rename @search.score -> search_score.
    """
    out: Dict[str, Any] = {}
    for k in ALLOWED_CANDIDATE_FIELDS:
        if k in candidate and candidate[k] is not None:
            if k == "@search.score":
                out["search_score"] = candidate[k]
            else:
                out[k] = candidate[k]
    return out


def fetch_candidates_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch candidates using Azure Search based on the request data
    """
    request = state["request"]

    try:
        search_result = search_student_by_query(request)
        candidates = search_result.get("results", [])

        print(f"DEBUG: Found {len(candidates)} candidates from Azure Search")

        # Debug print candidate information (excluding embeddings/vectors)
        for i, c in enumerate(candidates, 1):
            debug_candidate = {
                k: v for k, v in c.items()
                if not k.lower().endswith("embedding") and not k.lower().endswith("vector")
            }
            print(f"\nDEBUG: Candidate {i} details:")
            print(f"  Student ID: {c.get('student_id')}")
            print(f"  PEN: {c.get('pen')}")
            print(f"  Name: {c.get('legalFirstName')} {c.get('legalMiddleNames', '')} {c.get('legalLastName')}")
            print(f"  DOB: {c.get('dob')}")
            print(f"  Sex: {c.get('sexCode')}")
            print(f"  Postal Code: {c.get('postalCode')}")
            print(f"  Mincode: {c.get('mincode')}")
            print(f"  Local ID: {c.get('localID')}")
            print(f"  Grade: {c.get('gradeCode')}")
            print(f"  Full data: {debug_candidate}")

        return {
            "candidates": candidates,
            "search_metadata": {
                "status": search_result.get("status"),
                "search_type": search_result.get("search_type"),
                "count": search_result.get("count", 0),
                "pen_status": search_result.get("pen_status"),
            },
        }
    except Exception as e:
        print(f"Error fetching candidates: {e}")
        return {
            "candidates": [],
            "search_metadata": {"status": "error", "error": str(e)},
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
                "chosen_candidate": None,
                "mismatches": [],
                "ranked_candidates": [],
                "suspected_input_issues": [{"field": "unknown", "issue": "missing", "hint": "No candidates returned; check core identifiers (name, DOB, pen)."}],
            },
            "route": "end",
        }

    # Optional: you can still keep auto-confirm for 1 candidate,
    # but you asked: "if one match: return full info (with pen)" -> do it here.
    if candidate_count == 1:
        print("DEBUG: Single candidate found - auto-confirming without LLM")
        c = candidates[0]
        chosen = slim_candidate(c)

        return {
            "final_decision": "CONFIRM",
            "selected_candidate": c.get("student_id"),
            "confidence": 1.0,
            "llm_used": False,
            "analysis": {
                "decision": "CONFIRM",
                "confidence": 1.0,
                "reasons": ["Single candidate returned from search"],
                "chosen_candidate": chosen,   # full record echo
                "mismatches": [],
                "ranked_candidates": [],
                "suspected_input_issues": [],
            },
            "route": "end",
        }

    print(f"DEBUG: Multiple candidates ({candidate_count}) found - routing to LLM analysis")
    return {"route": "llm_analyze", "llm_used": True}


def llm_analyze_node(state: Dict[str, Any], llm_client: LLMClient) -> Dict[str, Any]:
    """
    Analyze candidates using LLM to make final decision
    Only called when multiple candidates are found
    """
    request = state["request"]
    candidates = state["candidates"]

    print(f"DEBUG: LLM Analysis starting with {len(candidates)} candidates")

    # Slim candidates: keep only the fields that schema allows (prevents schema/token issues)
    candidates_for_llm: List[Dict[str, Any]] = [slim_candidate(c) for c in candidates[:20]]

    print(f"DEBUG: Prepared {len(candidates_for_llm)} candidates for LLM analysis")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        request_json=json.dumps(request, ensure_ascii=False, indent=2),
        candidates_json=json.dumps(candidates_for_llm, ensure_ascii=False, indent=2),
    )

    print(f"DEBUG: Sending prompt to LLM (user prompt length: {len(user_prompt)} chars)")

    try:
        result = llm_client.with_structured_output_and_system(CandidateAnalysis).invoke(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,   # for gpt-4o-mini this is fine
        )

        print(f"DEBUG: LLM returned decision: {result.decision} with confidence: {result.confidence}")

        # If CONFIRM, set selected_candidate from chosen_candidate.student_id
        selected_id = None
        if result.chosen_candidate is not None:
            selected_id = result.chosen_candidate.student_id if result.chosen_candidate else None

        return {
            "analysis": result.model_dump(),
            "final_decision": result.decision,
            "selected_candidate": selected_id,
            "confidence": result.confidence,
            "llm_used": True,
            "route": "end",
        }

    except Exception as e:
        print(f"Error in LLM analysis: {e}")
        return {
            "analysis": {
                "decision": "NO_MATCH",
                "confidence": 0.0,
                "reasons": [f"LLM analysis failed: {str(e)}"],
                "chosen_candidate": None,
                "mismatches": [],
                "ranked_candidates": [],
                "suspected_input_issues": [{"field": "unknown", "issue": "conflict", "hint": "LLM call failed; check model deployment/base_url/schema."}],
            },
            "final_decision": "NO_MATCH",
            "selected_candidate": None,
            "confidence": 0.0,
            "llm_used": True,
            "error": str(e),
            "route": "end",
        }
