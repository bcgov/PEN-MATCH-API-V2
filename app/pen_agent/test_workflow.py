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

def test_pen_match_workflow():
    """Test the complete workflow with sample data"""
    
    # Updated test request format
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
    print(f"Test Request: {json.dumps(test_request, indent=2)}")
    print("-" * 50)
    
    try:
        # Create workflow instance
        workflow = create_pen_match_workflow()
        print("✓ Workflow created successfully")
        
        # Run the match
        result = workflow.run_match(test_request)
        print("✓ Workflow executed successfully")
        
        # Display results
        print(f"\nResults:")
        print(f"Success: {result.get('success')}")
        print(f"Final Decision: {result.get('final_decision')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Candidates Found: {result.get('candidates_count', 0)}")
        print(f"Model Used: {result.get('model_used')}")
        print(f"LLM Used: {result.get('llm_used', 'No')}")
        
        if result.get('error'):
            print(f"Error: {result.get('error')}")
        
        if result.get('search_metadata'):
            print(f"\nSearch Metadata:")
            print(json.dumps(result.get('search_metadata'), indent=2))
        
        if result.get('analysis'):
            analysis = result.get('analysis')
            print(f"\nAnalysis Details:")
            print(f"Decision: {analysis.get('decision')}")
            print(f"Reasons: {analysis.get('reasons', [])}")
            print(f"Mismatches: {analysis.get('mismatches', {})}")
        
        if result.get('candidates'):
            print(f"\nCandidates Debug Info:")
            for i, candidate in enumerate(result.get('candidates', []), 1):
                print(f"Candidate {i}:")
                print(json.dumps(candidate, indent=2))
                print("-" * 30)
        
        return result
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_pen_match_workflow()