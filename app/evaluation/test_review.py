import json
import sys
import os
from typing import Dict, List, Any, Tuple
import time

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pen_agent.workflow import create_pen_match_workflow

def load_test_queries_by_students(file_path: str, max_students: int = 10) -> List[Dict[str, Any]]:
    """Load test queries for the first N students from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            queries = data.get('queries', [])
            
        if not queries:
            return []
        
        # Since queries are already in order, we can just iterate and track unique PENs
        selected_queries = []
        seen_pens = set()
        students_processed = 0
        
        for query in queries:
            pen = query.get('ground_truth_pen', '')
            if pen not in seen_pens:
                seen_pens.add(pen)
                students_processed += 1
                
            if students_processed <= max_students:
                selected_queries.append(query)
            else:
                break
        
        return selected_queries
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return []
    except Exception as e:
        print(f"Error loading test queries: {e}")
        return []

def load_test_queries(file_path: str, max_queries: int = 100) -> List[Dict[str, Any]]:
    """Load test queries from JSON file (legacy function for backward compatibility)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            queries = data.get('queries', [])
            return queries[:max_queries]
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return []
    except Exception as e:
        print(f"Error loading test queries: {e}")
        return []

def convert_query_format(query_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert query format from test format to PEN agent format
    Test format uses: legalFirstName, legalLastName, legalMiddleNames, dob, etc.
    Agent format uses: givenName, surname, middleName, dob, etc.
    """
    converted = {}
    
    # Map field names
    field_mapping = {
        'legalFirstName': 'givenName',
        'legalLastName': 'surname', 
        'legalMiddleNames': 'middleName',
        'dob': 'dob',
        'sexCode': 'sexCode',
        'postalCode': 'postalCode',
        'mincode': 'mincode',
        'gradeCode': 'gradeCode',
        'localID': 'localID'
    }
    
    for test_field, agent_field in field_mapping.items():
        if test_field in query_data:
            value = query_data[test_field]
            if value and value != "":  # Only include non-empty values
                converted[agent_field] = value
    
    return converted

def map_agent_decision_to_label(agent_decision: str) -> str:
    """
    Map agent decision to expected review label
    Agent returns: CONFIRM, REVIEW, NO_MATCH
    Expected labels: CONFIRM, REVIEW
    """
    decision_mapping = {
        'CONFIRM': 'CONFIRM',
        'REVIEW': 'REVIEW', 
        'NO_MATCH': 'REVIEW'  # NO_MATCH should be treated as REVIEW for evaluation
    }
    return decision_mapping.get(agent_decision, 'UNKNOWN')

def run_single_test(query_info: Dict[str, Any], workflow) -> Dict[str, Any]:
    """Run a single test query through the PEN agent workflow"""
    try:
        query_data = query_info.get('query', {})
        expected_label = query_info.get('review_label', '')
        query_type = query_info.get('query_type', '')
        ground_truth_pen = query_info.get('ground_truth_pen', '')
        
        # Convert query format for the agent
        agent_query = convert_query_format(query_data)
        
        # Run through PEN agent workflow
        start_time = time.time()
        result = workflow.run_match(agent_query)
        processing_time = time.time() - start_time
        
        # Extract agent decision
        if result.get('success', False):
            agent_decision = result.get('final_decision', 'NO_MATCH')
            confidence = result.get('confidence', 0.0)
            candidates_count = result.get('candidates_count', 0)
            selected_candidate = result.get('selected_candidate')
            analysis = result.get('analysis', {})
        else:
            agent_decision = 'ERROR'
            confidence = 0.0
            candidates_count = 0
            selected_candidate = None
            analysis = {'error': result.get('error', 'Unknown error')}
        
        # Map to expected format
        predicted_label = map_agent_decision_to_label(agent_decision)
        
        # Check if prediction is correct
        correct_prediction = predicted_label == expected_label
        
        # Check if correct PEN was selected (for CONFIRM cases)
        correct_pen_selected = False
        if agent_decision == 'CONFIRM' and selected_candidate:
            # Extract PEN from selected candidate if available
            candidates = result.get('candidates', [])
            for candidate in candidates:
                if candidate.get('student_id') == selected_candidate or candidate.get('pen') == ground_truth_pen:
                    correct_pen_selected = True
                    break
        
        return {
            'success': True,
            'query_type': query_type,
            'expected_label': expected_label,
            'agent_decision': agent_decision,
            'predicted_label': predicted_label,
            'correct_prediction': correct_prediction,
            'correct_pen_selected': correct_pen_selected,
            'confidence': confidence,
            'candidates_count': candidates_count,
            'selected_candidate': selected_candidate,
            'ground_truth_pen': ground_truth_pen,
            'processing_time': processing_time,
            'original_query': query_data,
            'agent_query': agent_query,
            'analysis_reasons': analysis.get('reasons', []),
            'analysis_mismatches': analysis.get('mismatches', [])
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'query_type': query_info.get('query_type', 'Unknown'),
            'expected_label': query_info.get('review_label', 'Unknown'),
            'agent_decision': 'ERROR',
            'predicted_label': 'ERROR',
            'correct_prediction': False,
            'correct_pen_selected': False,
            'confidence': 0.0,
            'candidates_count': 0,
            'processing_time': 0.0
        }

def evaluate_agent_decisions(test_file_path: str = "app/evaluation/test_queries_review.json", max_students: int = 10) -> Dict[str, Any]:
    """Evaluate PEN agent decisions against expected review labels"""
    
    print("Loading test queries...")
    test_queries = load_test_queries_by_students(test_file_path, max_students)
    
    if not test_queries:
        return None
    
    # Count unique students for reporting
    unique_pens = set(query.get('ground_truth_pen', '') for query in test_queries)
    
    print(f"Loaded {len(test_queries)} test queries for {len(unique_pens)} students")
    print("Initializing PEN agent workflow...")
    
    # Initialize the workflow
    try:
        workflow = create_pen_match_workflow()
    except Exception as e:
        print(f"Failed to initialize PEN agent workflow: {e}")
        return None
    
    print("Starting evaluation...")
    
    results = []
    total_queries = len(test_queries)
    
    # Progress tracking
    for i, query_info in enumerate(test_queries, 1):
        if i % 10 == 0 or i == 1:
            print(f"Processing query {i}/{total_queries} ({i/total_queries*100:.1f}%)")
        
        result = run_single_test(query_info, workflow)
        results.append(result)
    
    print(f"\nCompleted evaluation of {len(results)} queries for {len(unique_pens)} students")
    return analyze_results(results)

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the evaluation results and calculate metrics"""
    
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]
    
    if not successful_tests:
        print("No successful tests to analyze!")
        return None
    
    # Overall accuracy
    correct_predictions = [r for r in successful_tests if r['correct_prediction']]
    overall_accuracy = len(correct_predictions) / len(successful_tests)
    
    # Accuracy by expected label
    label_stats = {}
    for expected_label in ['CONFIRM', 'REVIEW']:
        label_tests = [r for r in successful_tests if r['expected_label'] == expected_label]
        if label_tests:
            label_correct = [r for r in label_tests if r['correct_prediction']]
            label_stats[expected_label] = {
                'total': len(label_tests),
                'correct': len(label_correct),
                'accuracy': len(label_correct) / len(label_tests),
                'avg_confidence': sum(r['confidence'] for r in label_tests) / len(label_tests),
                'avg_candidates': sum(r['candidates_count'] for r in label_tests) / len(label_tests)
            }
    
    # Accuracy by query type
    query_type_stats = {}
    for result in successful_tests:
        qtype = result['query_type']
        if qtype not in query_type_stats:
            query_type_stats[qtype] = {
                'total': 0,
                'correct': 0,
                'confirm_predicted': 0,
                'review_predicted': 0
            }
        
        stats = query_type_stats[qtype]
        stats['total'] += 1
        if result['correct_prediction']:
            stats['correct'] += 1
        
        if result['predicted_label'] == 'CONFIRM':
            stats['confirm_predicted'] += 1
        elif result['predicted_label'] == 'REVIEW':
            stats['review_predicted'] += 1
    
    # Calculate accuracy for each query type
    for qtype, stats in query_type_stats.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    
    # Agent decision distribution
    decision_distribution = {}
    for result in successful_tests:
        decision = result['agent_decision']
        decision_distribution[decision] = decision_distribution.get(decision, 0) + 1
    
    # Confusion matrix
    confusion_matrix = {}
    for result in successful_tests:
        expected = result['expected_label']
        predicted = result['predicted_label']
        
        if expected not in confusion_matrix:
            confusion_matrix[expected] = {}
        if predicted not in confusion_matrix[expected]:
            confusion_matrix[expected][predicted] = 0
        confusion_matrix[expected][predicted] += 1
    
    # PEN selection accuracy (for CONFIRM cases only)
    confirm_tests = [r for r in successful_tests if r['expected_label'] == 'CONFIRM']
    pen_selection_accuracy = 0.0
    if confirm_tests:
        correct_pen_selections = [r for r in confirm_tests if r['correct_pen_selected']]
        pen_selection_accuracy = len(correct_pen_selections) / len(confirm_tests)
    
    # Performance metrics
    avg_processing_time = sum(r['processing_time'] for r in successful_tests) / len(successful_tests)
    
    return {
        'evaluation_summary': {
            'total_queries': len(results),
            'successful_tests': len(successful_tests),
            'failed_tests': len(failed_tests),
            'overall_accuracy': overall_accuracy,
            'pen_selection_accuracy': pen_selection_accuracy,
            'avg_processing_time': avg_processing_time
        },
        'accuracy_by_label': label_stats,
        'accuracy_by_query_type': query_type_stats,
        'agent_decision_distribution': decision_distribution,
        'confusion_matrix': confusion_matrix,
        'detailed_results': results
    }

def print_evaluation_report(analysis: Dict[str, Any]):
    """Print a comprehensive evaluation report"""
    
    print("\n" + "="*80)
    print("PEN AGENT REVIEW LABEL EVALUATION REPORT")
    print("="*80)
    
    # Summary
    summary = analysis['evaluation_summary']
    print(f"\nEVALUATION SUMMARY:")
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Successful Tests: {summary['successful_tests']}")
    print(f"Failed Tests: {summary['failed_tests']}")
    print(f"Overall Accuracy: {summary['overall_accuracy']:.2%}")
    print(f"PEN Selection Accuracy (CONFIRM cases): {summary['pen_selection_accuracy']:.2%}")
    print(f"Average Processing Time: {summary['avg_processing_time']:.3f} seconds")
    
    # Accuracy by label
    print(f"\nACCURACY BY EXPECTED LABEL:")
    for label, stats in analysis['accuracy_by_label'].items():
        print(f"\n{label}:")
        print(f"  Total Tests: {stats['total']}")
        print(f"  Correct: {stats['correct']}")
        print(f"  Accuracy: {stats['accuracy']:.2%}")
        print(f"  Avg Confidence: {stats['avg_confidence']:.3f}")
        print(f"  Avg Candidates: {stats['avg_candidates']:.1f}")
    
    # Accuracy by query type
    print(f"\nACCURACY BY QUERY TYPE:")
    for qtype, stats in sorted(analysis['accuracy_by_query_type'].items()):
        print(f"\n{qtype}:")
        print(f"  Total: {stats['total']}")
        print(f"  Correct: {stats['correct']}")
        print(f"  Accuracy: {stats['accuracy']:.2%}")
        print(f"  CONFIRM Predicted: {stats['confirm_predicted']}")
        print(f"  REVIEW Predicted: {stats['review_predicted']}")
    
    # Agent decision distribution
    print(f"\nAGENT DECISION DISTRIBUTION:")
    for decision, count in analysis['agent_decision_distribution'].items():
        percentage = count / summary['successful_tests'] * 100
        print(f"  {decision}: {count} ({percentage:.1f}%)")
    
    # Confusion matrix
    print(f"\nCONFUSION MATRIX:")
    print(f"{'Expected \\ Predicted':<20} {'CONFIRM':<10} {'REVIEW':<10} {'ERROR':<10}")
    print("-" * 50)
    for expected, predictions in analysis['confusion_matrix'].items():
        row = f"{expected:<20}"
        for predicted in ['CONFIRM', 'REVIEW', 'ERROR']:
            count = predictions.get(predicted, 0)
            row += f"{count:<10}"
        print(row)

def save_results(analysis: Dict[str, Any], output_file: str = "app/evaluation/agent_evaluation_results.json"):
    """Save evaluation results to JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """Main evaluation function"""
    print("PEN Agent Review Label Evaluation")
    print("-" * 40)
    
    # Configuration
    test_file = "app/evaluation/test_queries_review.json"
    max_students = 10  # Test first 10 students instead of 100 queries
    
    print(f"Test file: {test_file}")
    print(f"Max students: {max_students}")
    
    # Run evaluation
    start_time = time.time()
    analysis = evaluate_agent_decisions(test_file, max_students)
    total_time = time.time() - start_time
    
    if analysis:
        print(f"\nTotal evaluation time: {total_time:.2f} seconds")
        
        # Print report
        print_evaluation_report(analysis)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"app/evaluation/agent_evaluation_results_{timestamp}.json"
        save_results(analysis, output_file)
        
        # Quick summary for easy reference
        print(f"\n" + "="*50)
        print(f"QUICK SUMMARY")
        print(f"="*50)
        print(f"Overall Accuracy: {analysis['evaluation_summary']['overall_accuracy']:.2%}")
        print(f"PEN Selection Accuracy: {analysis['evaluation_summary']['pen_selection_accuracy']:.2%}")
        
        # Show accuracy breakdown
        confirm_acc = analysis['accuracy_by_label'].get('CONFIRM', {}).get('accuracy', 0)
        review_acc = analysis['accuracy_by_label'].get('REVIEW', {}).get('accuracy', 0)
        print(f"CONFIRM Accuracy: {confirm_acc:.2%}")
        print(f"REVIEW Accuracy: {review_acc:.2%}")
        print(f"="*50)
    
    else:
        print("Evaluation failed!")

if __name__ == "__main__":
    main()