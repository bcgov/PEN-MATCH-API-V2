import json
import sys
import os
from typing import Dict, List, Any, Tuple
import time

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure_search.azure_search_query import search_student_by_query

def load_test_queries(file_path: str) -> Dict[str, Any]:
    """Load test queries from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return None
    except Exception as e:
        print(f"Error loading test queries: {e}")
        return None

def extract_pen_from_results(results: List[Dict[str, Any]]) -> List[str]:
    """Extract PEN numbers from search results"""
    pens = []
    for result in results:
        pen = result.get('pen', '')
        if pen:
            pens.append(str(pen))
    return pens

def calculate_recall_at_k(ground_truth_pen: str, retrieved_pens: List[str], k: int = 20) -> float:
    """
    Calculate recall@k
    Recall@k = 1 if ground truth is in top-k results, 0 otherwise
    """
    top_k_pens = retrieved_pens[:k]
    return 1.0 if ground_truth_pen in top_k_pens else 0.0

def calculate_mrr(ground_truth_pen: str, retrieved_pens: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR)
    MRR = 1/rank if ground truth is found, 0 otherwise
    """
    try:
        rank = retrieved_pens.index(ground_truth_pen) + 1  # 1-indexed
        return 1.0 / rank
    except ValueError:
        return 0.0

def run_single_query(query_data: Dict[str, Any], ground_truth_pen: str, query_info: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single query and evaluate results"""
    try:
        # Execute search query
        start_time = time.time()
        search_result = search_student_by_query(query_data)
        search_time = time.time() - start_time
        
        # Extract results
        results = search_result.get('results', [])
        pen_status = search_result.get('pen_status', 'Unknown')
        search_type = search_result.get('search_type', 'Unknown')
        count = search_result.get('count', 0)
        
        # Get top 20 candidates (or all if less than 20)
        top_20_results = results[:20]
        retrieved_pens = extract_pen_from_results(top_20_results)
        
        # Calculate metrics
        recall_20 = calculate_recall_at_k(ground_truth_pen, retrieved_pens, 20)
        mrr = calculate_mrr(ground_truth_pen, retrieved_pens)
        
        # Find rank of ground truth
        rank = None
        if ground_truth_pen in retrieved_pens:
            rank = retrieved_pens.index(ground_truth_pen) + 1
        
        return {
            'success': True,
            'query_type': query_info.get('query_type', 'Unknown'),
            'review_label': query_info.get('review_label', 'Unknown'),
            'ground_truth_pen': ground_truth_pen,
            'pen_status': pen_status,
            'search_type': search_type,
            'total_count': count,
            'top_20_count': len(top_20_results),
            'retrieved_pens': retrieved_pens,
            'recall_20': recall_20,
            'mrr': mrr,
            'rank': rank,
            'search_time': search_time,
            'found_in_top_20': recall_20 == 1.0
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'query_type': query_info.get('query_type', 'Unknown'),
            'review_label': query_info.get('review_label', 'Unknown'),
            'ground_truth_pen': ground_truth_pen,
            'recall_20': 0.0,
            'mrr': 0.0,
            'rank': None,
            'search_time': 0.0,
            'found_in_top_20': False
        }

def run_evaluation(test_file_path: str = "test_queries_review.json", max_queries: int = None) -> Dict[str, Any]:
    """Run evaluation on all test queries"""
    
    print("Loading test queries...")
    test_data = load_test_queries(test_file_path)
    if not test_data:
        return None
    
    queries = test_data.get('queries', [])
    if max_queries:
        queries = queries[:max_queries]
    
    print(f"Loaded {len(queries)} test queries")
    print("Starting evaluation...")
    
    results = []
    total_queries = len(queries)
    
    # Progress tracking
    for i, query_info in enumerate(queries, 1):
        if i % 10 == 0 or i == 1:
            print(f"Processing query {i}/{total_queries} ({i/total_queries*100:.1f}%)")
        
        query_data = query_info.get('query', {})
        ground_truth_pen = str(query_info.get('ground_truth_pen', ''))
        
        if not query_data or not ground_truth_pen:
            print(f"Warning: Skipping invalid query {i}")
            continue
        
        result = run_single_query(query_data, ground_truth_pen, query_info)
        results.append(result)
    
    print(f"\nCompleted evaluation of {len(results)} queries")
    return analyze_results(results, test_data.get('metadata', {}))

def analyze_results(results: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze evaluation results and calculate metrics"""
    
    successful_queries = [r for r in results if r.get('success', False)]
    failed_queries = [r for r in results if not r.get('success', False)]
    
    if not successful_queries:
        print("No successful queries to analyze!")
        return None
    
    # Overall metrics
    total_recall_20 = sum(r['recall_20'] for r in successful_queries)
    total_mrr = sum(r['mrr'] for r in successful_queries)
    
    avg_recall_20 = total_recall_20 / len(successful_queries)
    avg_mrr = total_mrr / len(successful_queries)
    
    # Metrics by query type
    query_type_stats = {}
    for result in successful_queries:
        qtype = result['query_type']
        if qtype not in query_type_stats:
            query_type_stats[qtype] = {
                'count': 0,
                'recall_20_sum': 0,
                'mrr_sum': 0,
                'found_count': 0
            }
        
        stats = query_type_stats[qtype]
        stats['count'] += 1
        stats['recall_20_sum'] += result['recall_20']
        stats['mrr_sum'] += result['mrr']
        if result['found_in_top_20']:
            stats['found_count'] += 1
    
    # Calculate averages for each query type
    for qtype, stats in query_type_stats.items():
        stats['avg_recall_20'] = stats['recall_20_sum'] / stats['count']
        stats['avg_mrr'] = stats['mrr_sum'] / stats['count']
        stats['success_rate'] = stats['found_count'] / stats['count']
    
    # Metrics by review label (CONFIRM vs REVIEW)
    label_stats = {}
    for result in successful_queries:
        label = result['review_label']
        if label not in label_stats:
            label_stats[label] = {
                'count': 0,
                'recall_20_sum': 0,
                'mrr_sum': 0,
                'found_count': 0
            }
        
        stats = label_stats[label]
        stats['count'] += 1
        stats['recall_20_sum'] += result['recall_20']
        stats['mrr_sum'] += result['mrr']
        if result['found_in_top_20']:
            stats['found_count'] += 1
    
    # Calculate averages for each label
    for label, stats in label_stats.items():
        stats['avg_recall_20'] = stats['recall_20_sum'] / stats['count']
        stats['avg_mrr'] = stats['mrr_sum'] / stats['count']
        stats['success_rate'] = stats['found_count'] / stats['count']
    
    # Search type distribution
    search_type_dist = {}
    pen_status_dist = {}
    
    for result in successful_queries:
        stype = result.get('search_type', 'Unknown')
        search_type_dist[stype] = search_type_dist.get(stype, 0) + 1
        
        pstatus = result.get('pen_status', 'Unknown')
        pen_status_dist[pstatus] = pen_status_dist.get(pstatus, 0) + 1
    
    # Rank distribution for found items
    rank_distribution = {}
    found_results = [r for r in successful_queries if r['found_in_top_20']]
    for result in found_results:
        rank = result['rank']
        if rank:
            rank_bin = f"Rank {rank}"
            rank_distribution[rank_bin] = rank_distribution.get(rank_bin, 0) + 1
    
    analysis = {
        'evaluation_summary': {
            'total_queries': len(results),
            'successful_queries': len(successful_queries),
            'failed_queries': len(failed_queries),
            'success_rate': len(successful_queries) / len(results) if results else 0
        },
        'overall_metrics': {
            'recall_at_20': avg_recall_20,
            'mean_reciprocal_rank': avg_mrr,
            'queries_found_in_top_20': len(found_results),
            'percentage_found': len(found_results) / len(successful_queries) * 100 if successful_queries else 0
        },
        'metrics_by_query_type': query_type_stats,
        'metrics_by_review_label': label_stats,
        'search_type_distribution': search_type_dist,
        'pen_status_distribution': pen_status_dist,
        'rank_distribution': rank_distribution,
        'detailed_results': results,
        'metadata': metadata
    }
    
    return analysis

def print_evaluation_report(analysis: Dict[str, Any]):
    """Print a comprehensive evaluation report"""
    
    print("\n" + "="*80)
    print("AZURE SEARCH EVALUATION REPORT")
    print("="*80)
    
    # Summary
    summary = analysis['evaluation_summary']
    print(f"\nEVALUATION SUMMARY:")
    print(f"Total Queries: {summary['total_queries']}")
    print(f"Successful Queries: {summary['successful_queries']}")
    print(f"Failed Queries: {summary['failed_queries']}")
    print(f"Success Rate: {summary['success_rate']:.2%}")
    
    # Overall metrics
    metrics = analysis['overall_metrics']
    print(f"\nOVERALL METRICS:")
    print(f"Recall@20: {metrics['recall_at_20']:.4f}")
    print(f"Mean Reciprocal Rank (MRR): {metrics['mean_reciprocal_rank']:.4f}")
    print(f"Queries Found in Top 20: {metrics['queries_found_in_top_20']} ({metrics['percentage_found']:.1f}%)")
    
    # Metrics by review label
    print(f"\nMETRICS BY REVIEW LABEL:")
    for label, stats in analysis['metrics_by_review_label'].items():
        print(f"\n{label}:")
        print(f"  Queries: {stats['count']}")
        print(f"  Recall@20: {stats['avg_recall_20']:.4f}")
        print(f"  MRR: {stats['avg_mrr']:.4f}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
    
    # Metrics by query type
    print(f"\nMETRICS BY QUERY TYPE:")
    for qtype, stats in sorted(analysis['metrics_by_query_type'].items()):
        print(f"\n{qtype}:")
        print(f"  Queries: {stats['count']}")
        print(f"  Recall@20: {stats['avg_recall_20']:.4f}")
        print(f"  MRR: {stats['avg_mrr']:.4f}")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
    
    # Search type distribution
    print(f"\nSEARCH TYPE DISTRIBUTION:")
    for stype, count in analysis['search_type_distribution'].items():
        percentage = count / summary['successful_queries'] * 100 if summary['successful_queries'] else 0
        print(f"  {stype}: {count} ({percentage:.1f}%)")
    
    # PEN status distribution
    print(f"\nPEN STATUS DISTRIBUTION:")
    for pstatus, count in analysis['pen_status_distribution'].items():
        percentage = count / summary['successful_queries'] * 100 if summary['successful_queries'] else 0
        print(f"  {pstatus}: {count} ({percentage:.1f}%)")
    
    # Top rank positions for found items
    if analysis['rank_distribution']:
        print(f"\nRANK DISTRIBUTION (Found Items Only):")
        sorted_ranks = sorted(analysis['rank_distribution'].items(), 
                            key=lambda x: int(x[0].split()[1]))
        for rank, count in sorted_ranks[:10]:  # Show top 10 ranks
            print(f"  {rank}: {count}")
        if len(sorted_ranks) > 10:
            print(f"  ... and {len(sorted_ranks) - 10} more ranks")

def save_results(analysis: Dict[str, Any], output_file: str = "evaluation_results.json"):
    """Save evaluation results to JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """Main evaluation function"""
    print("Azure Search Retrieval Evaluation")
    print("-" * 40)
    
    # Configuration
    test_file = "test_queries_review.json"
    max_queries = None  # Set to a number to limit queries for testing, None for all
    
    print(f"Test file: {test_file}")
    if max_queries:
        print(f"Max queries: {max_queries}")
    
    # Run evaluation
    start_time = time.time()
    analysis = run_evaluation(test_file, max_queries)
    total_time = time.time() - start_time
    
    if analysis:
        print(f"\nTotal evaluation time: {total_time:.2f} seconds")
        print(f"Average time per query: {total_time/analysis['evaluation_summary']['total_queries']:.3f} seconds")
        
        # Print report
        print_evaluation_report(analysis)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_results_{timestamp}.json"
        save_results(analysis, output_file)
        
        # Quick summary for easy reference
        print(f"\n" + "="*50)
        print(f"QUICK SUMMARY")
        print(f"="*50)
        print(f"Recall@20: {analysis['overall_metrics']['recall_at_20']:.4f}")
        print(f"MRR: {analysis['overall_metrics']['mean_reciprocal_rank']:.4f}")
        print(f"Found in Top 20: {analysis['overall_metrics']['percentage_found']:.1f}%")
        print(f"="*50)
    
    else:
        print("Evaluation failed!")

if __name__ == "__main__":
    main()