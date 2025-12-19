# pen_agent/nodes.py
from typing import Dict, Any, List
import json

from azure_search.azure_search_query import search_student_by_query
from .schemas import CandidateAnalysis
from .prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .llm_client import LLMClient


# Core fields we always pass (and that schema supports)
CORE_FIELDS = {
    "student_id", "pen",
    "legalFirstName", "legalMiddleNames", "legalLastName",
    "dob", "sexCode", "mincode", "postalCode",
    "localID", "gradeCode",
    "@search.score", "final_score", "search_method",
}

RENAME_MAP = {"@search.score": "search_score"}

# Keep extras small to avoid huge prompts (24k+ chars)
# Add fields you actually want visible as reference/debug
EXTRA_ALLOWLIST = {
    "base_search_score",
    "dob_sim",
    "mincode_sim",
    "postal_sim",
    "sex_sim",
    "pen_status",
    "penStatus",
    "@search.reranker_score",
}

MAX_EXTRAS_PER_CANDIDATE = 12


def to_candidate_payload(candidate: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """
    Convert raw Azure Search candidate dict into:
    - core fields (schema-defined)
    - extras: allowlisted keys only, stringified
    Removes embedding/vector fields.
    """
    out: Dict[str, Any] = {"rank": rank}
    extras: List[Dict[str, Any]] = []

    for k, v in candidate.items():
        lk = k.lower()
        if lk.endswith("embedding") or lk.endswith("vector"):
            continue

        if k in CORE_FIELDS:
            out[RENAME_MAP.get(k, k)] = v
            continue

        # only keep selected extras to reduce prompt length
        if k not in EXTRA_ALLOWLIST:
            continue

        if v is None:
            extras.append({"key": k, "value": None})
        elif isinstance(v, (str, int, float, bool)):
            extras.append({"key": k, "value": str(v)})
        else:
            extras.append({"key": k, "value": json.dumps(v, ensure_ascii=False)})

    if "student_id" not in out:
        out["student_id"] = str(candidate.get("student_id", ""))

    out["extras"] = extras[:MAX_EXTRAS_PER_CANDIDATE]
    return out


def fetch_candidates_node(state: Dict[str, Any]) -> Dict[str, Any]:
    request = state["request"]
    try:
        search_result = search_student_by_query(request)
        candidates = search_result.get("results", [])
        print(f"DEBUG: Found {len(candidates)} candidates from Azure Search")

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
        return {"candidates": [], "search_metadata": {"status": "error", "error": str(e)}}


def decision_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    candidates = state["candidates"]
    n = len(candidates)
    print(f"DEBUG: Decision router - found {n} candidates")

    if n == 0:
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
                "review_candidates": [],
                "suspected_input_issues": [
                    {"field": "unknown", "issue": "missing", "hint": "No candidates returned; re-check name/DOB/PEN/mincode/postal."}
                ],
            },
            "route": "end",
        }

    if n == 1:
        chosen = to_candidate_payload(candidates[0], rank=1)
        return {
            "final_decision": "CONFIRM",
            "selected_candidate": chosen.get("student_id"),
            "confidence": 1.0,
            "llm_used": False,
            "analysis": {
                "decision": "CONFIRM",
                "confidence": 1.0,
                "reasons": ["Single candidate returned from search"],
                "chosen_candidate": chosen,
                "mismatches": [],
                "review_candidates": [],
                "suspected_input_issues": [],
            },
            "route": "end",
        }

    return {"route": "llm_analyze", "llm_used": True}


def llm_analyze_node(state: Dict[str, Any], llm_client: LLMClient) -> Dict[str, Any]:
    request = state["request"]
    candidates = state["candidates"]

    candidates_for_llm = [to_candidate_payload(c, rank=i + 1) for i, c in enumerate(candidates[:20])]

    candidates_json_str = json.dumps(candidates_for_llm, ensure_ascii=False, indent=2)
    print("DEBUG: candidates_json chars =", len(candidates_json_str))
    print(
        "DEBUG: avg extras per candidate =",
        sum(len(c.get("extras", [])) for c in candidates_for_llm) / max(1, len(candidates_for_llm))
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        request_json=json.dumps(request, ensure_ascii=False, indent=2),
        candidates_json=candidates_json_str,
    )

    print(f"DEBUG: Sending prompt to LLM (user prompt length: {len(user_prompt)} chars)")

    try:
        result = llm_client.with_structured_output_and_system(CandidateAnalysis).invoke(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        dump = result.model_dump()
        print("DEBUG: review_candidates count =", len(dump.get("review_candidates", [])))
        if dump.get("review_candidates"):
            print("DEBUG: review_candidates[0] preview:")
            print(json.dumps(dump["review_candidates"][0], indent=2, ensure_ascii=False))

        selected_id = result.chosen_candidate.student_id if result.chosen_candidate else None

        return {
            "analysis": dump,
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
                "review_candidates": [],
                "suspected_input_issues": [
                    {"field": "unknown", "issue": "conflict", "hint": "LLM call failed; check model/base_url/schema/prompt."}
                ],
            },
            "final_decision": "NO_MATCH",
            "selected_candidate": None,
            "confidence": 0.0,
            "llm_used": True,
            "error": str(e),
            "route": "end",
        }
