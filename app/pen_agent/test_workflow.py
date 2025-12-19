"""
Test script for PEN Match Workflow
Run this to verify the workflow can produce analysis results
"""

import sys
import os
import json

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pen_agent.workflow import create_pen_match_workflow


def _print_kv_extras(extras, max_items: int = 12):
    if not extras:
        print("  Extras: []")
        return
    print(f"  Extras (showing up to {max_items}):")
    for kv in extras[:max_items]:
        print(f"    - {kv.get('key')}: {kv.get('value')}")
    if len(extras) > max_items:
        print(f"    ... ({len(extras) - max_items} more)")


def _print_candidate(candidate: dict, show_extras: bool = True, extras_max: int = 12):
    # candidate is the schema CandidateRecord dict
    print(f"  Rank: {candidate.get('rank')}")
    print(f"  student_id: {candidate.get('student_id')}")
    print(f"  pen: {candidate.get('pen')}")
    print(f"  Name: {candidate.get('legalFirstName')} {candidate.get('legalMiddleNames') or ''} {candidate.get('legalLastName')}")
    print(f"  DOB: {candidate.get('dob')}")
    print(f"  sexCode: {candidate.get('sexCode')}")
    print(f"  mincode: {candidate.get('mincode')}")
    print(f"  postalCode: {candidate.get('postalCode')}")
    print(f"  localID: {candidate.get('localID')}")
    print(f"  gradeCode: {candidate.get('gradeCode')}")
    print(f"  search_score: {candidate.get('search_score')}")
    print(f"  final_score: {candidate.get('final_score')}")
    print(f"  search_method: {candidate.get('search_method')}")

    if show_extras:
        _print_kv_extras(candidate.get("extras", []), max_items=extras_max)


def test_pen_match_workflow():
    """Test the complete workflow with sample data"""

    test_request = {
        "dob": "20100405",
        "sex": "",
        "surname": "LI",
        "pen": "",
        "givenName": "MICHAEL",
        "middleName": "",
        "mincode": "05757079",
        "localID": "",
        "postal": ""
    }

    print("Testing PEN Match Workflow...")
    print(f"Test Request:\n{json.dumps(test_request, indent=2)}")
    print("-" * 60)

    try:
        workflow = create_pen_match_workflow()
        print("✓ Workflow created successfully")

        result = workflow.run_match(test_request)
        print("✓ Workflow executed successfully")

        print("\nResults:")
        print(f"  Success: {result.get('success')}")
        print(f"  Final Decision: {result.get('final_decision')}")
        print(f"  Confidence: {result.get('confidence')}")
        print(f"  Candidates Found: {result.get('candidates_count', 0)}")
        print(f"  Model Used: {result.get('model_used')}")
        print(f"  LLM Used: {result.get('llm_used', False)}")

        if result.get("error"):
            print(f"\nError:\n  {result.get('error')}")

        if result.get("search_metadata"):
            print("\nSearch Metadata:")
            print(json.dumps(result.get("search_metadata"), indent=2))

        analysis = result.get("analysis") or {}
        if analysis:
            print("\nAnalysis Details:")
            print(f"  Decision: {analysis.get('decision')}")
            print(f"  Confidence: {analysis.get('confidence')}")
            print(f"  Reasons: {analysis.get('reasons', [])}")

            mismatches = analysis.get("mismatches", [])
            print(f"  Mismatches ({len(mismatches)}):")
            for m in mismatches:
                print(f"    - {m.get('field')} ({m.get('severity')}): {m.get('detail')}")

            decision = analysis.get("decision")

            # CONFIRM: print chosen candidate full info
            if decision == "CONFIRM":
                print("\nChosen Candidate (CONFIRM):")
                chosen = analysis.get("chosen_candidate")
                if not chosen:
                    print("  (chosen_candidate is missing!)")
                else:
                    _print_candidate(chosen, show_extras=True, extras_max=20)

            # REVIEW: print 5 candidates, each with its own reasons/issues
            elif decision == "REVIEW":
                review_list = analysis.get("review_candidates", [])
                # print(f"\nReview Candidates ({len(review_list)}):")
                # for idx, rc in enumerate(review_list, 1):
                #     cand = rc.get("candidate", {})
                #     print("\n" + "=" * 60)
                #     print(f"Candidate {idx}:")
                #     _print_candidate(cand, show_extras=True, extras_max=12)

                #     reasons = rc.get("reasons", [])
                #     issues = rc.get("issues", [])

                #     print(f"  Candidate Reasons ({len(reasons)}):")
                #     for r in reasons:
                #         print(f"    - {r}")

                #     print(f"  Candidate Issues ({len(issues)}):")
                #     for it in issues:
                #         print(f"    - {it.get('field')} ({it.get('severity')}): {it.get('detail')}")

                if len(review_list) == 0:
                    print("  (No review_candidates returned. Check prompt/schema compliance.)")

            # NO_MATCH: print suggested fields to check
            elif decision == "NO_MATCH":
                issues = analysis.get("suspected_input_issues", [])
                print(f"\nSuspected Input Issues ({len(issues)}):")
                for it in issues:
                    print(f"  - {it.get('field')} [{it.get('issue')}]: {it.get('hint')}")

        return result

    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_pen_match_workflow()
