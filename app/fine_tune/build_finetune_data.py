import json
import random
from pathlib import Path
from collections import defaultdict

SYSTEM = (
    "You are a PEN-MATCH decision classifier for BC student records. "
    "Given a STUDENT REQUEST, output JSON only as "
    "{\"decision\":\"CONFIRM\"} or {\"decision\":\"REVIEW\"}. "
    "CONFIRM: Use only when the request contains minor typos but clearly matches a student record. "
    "REVIEW: Use when there are conflicting information or significant discrepancies that require human review."
)

USER_TEMPLATE = "STUDENT REQUEST (JSON):\n{request}\n\nReturn JSON only."

def normalize_label(label: str) -> str:
    label = label.strip().upper()
    if label not in {"CONFIRM", "REVIEW"}:
        raise ValueError(f"Unexpected label: {label}")
    return label

def load_queries(test_query_path: Path):
    """Load queries from test_queries_review.json"""
    data = json.loads(test_query_path.read_text(encoding="utf-8"))
    queries = data.get("queries", [])
    if not queries:
        raise ValueError("No queries found in test_queries_review.json (missing 'queries').")
    return queries

def group_by_student_ordered(queries):
    """Group queries by student while preserving the original order of students"""
    student_order = []
    groups = {}
    
    for q in queries:
        pen = q.get("ground_truth_pen", "UNKNOWN_STUDENT")
        if pen not in groups:
            student_order.append(pen)
            groups[pen] = []
        groups[pen].append(q)
    
    # Return list of (pen, queries) tuples in original order
    return [(pen, groups[pen]) for pen in student_order]

def make_example(request_obj, label):
    """Create a fine-tuning example in the correct format"""
    request_str = json.dumps(request_obj, ensure_ascii=False, separators=(',', ':'))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TEMPLATE.format(request=request_str)},
            {"role": "assistant", "content": json.dumps({"decision": label})},
        ]
    }

def write_jsonl(path: Path, examples):
    """Write examples to JSONL format"""
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def build_examples_from_groups(groups):
    """Build training examples from grouped queries"""
    examples = []
    for pen, queries in groups:
        for q in queries:
            try:
                label = normalize_label(q["review_label"])
                request = q["query"]
                examples.append(make_example(request, label))
            except Exception as e:
                print(f"Warning: Skipping query for student {pen}: {e}")
                continue
    return examples

def main(
    input_path="app/fine_tune/test_queries_review.json",
    out_dir="finetune_data",
    test_students=10,
    val_students=10,
    seed=42
):
    """
    Generate fine-tuning data splits:
    - First 10 students: test
    - Next 10 students: validation  
    - Remaining students: train
    """
    random.seed(seed)
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading queries from {input_path}...")
    queries = load_queries(input_path)
    
    print("Grouping queries by student (preserving order)...")
    student_groups = group_by_student_ordered(queries)
    
    total_students = len(student_groups)
    print(f"Found {total_students} students with {len(queries)} total queries")
    
    if total_students < test_students + val_students:
        raise ValueError(f"Not enough students. Need at least {test_students + val_students}, got {total_students}")
    
    # Split by student groups (not shuffled to preserve order)
    test_groups = student_groups[:test_students]
    val_groups = student_groups[test_students:test_students + val_students] 
    train_groups = student_groups[test_students + val_students:]
    
    print(f"Split: {len(test_groups)} test students, {len(val_groups)} val students, {len(train_groups)} train students")
    
    # Build examples for each split
    print("Building test examples...")
    test_examples = build_examples_from_groups(test_groups)
    
    print("Building validation examples...")
    val_examples = build_examples_from_groups(val_groups)
    
    print("Building training examples...")
    train_examples = build_examples_from_groups(train_groups)
    
    # Write to files
    test_file = out_dir / "test.jsonl"
    val_file = out_dir / "validation.jsonl"
    train_file = out_dir / "train.jsonl"
    
    write_jsonl(test_file, test_examples)
    write_jsonl(val_file, val_examples)
    write_jsonl(train_file, train_examples)
    
    # Print statistics
    print("\n" + "="*50)
    print("FINE-TUNING DATA GENERATION COMPLETE")
    print("="*50)
    print(f"Input file: {input_path}")
    print(f"Output directory: {out_dir.resolve()}")
    print()
    print("DATA SPLITS:")
    print(f"Test examples: {len(test_examples)} (students 1-{test_students})")
    print(f"Validation examples: {len(val_examples)} (students {test_students+1}-{test_students+val_students})")
    print(f"Train examples: {len(train_examples)} (students {test_students+val_students+1}+)")
    print(f"Total examples: {len(test_examples) + len(val_examples) + len(train_examples)}")
    
    # Analyze label distribution
    def count_labels(examples):
        confirm_count = sum(1 for ex in examples if "CONFIRM" in ex["messages"][2]["content"])
        review_count = sum(1 for ex in examples if "REVIEW" in ex["messages"][2]["content"])
        return confirm_count, review_count
    
    train_confirm, train_review = count_labels(train_examples)
    val_confirm, val_review = count_labels(val_examples)
    test_confirm, test_review = count_labels(test_examples)
    
    print()
    print("LABEL DISTRIBUTION:")
    print(f"Train - CONFIRM: {train_confirm}, REVIEW: {train_review} (ratio: {train_confirm/(train_confirm+train_review):.2f})")
    print(f"Val   - CONFIRM: {val_confirm}, REVIEW: {val_review} (ratio: {val_confirm/(val_confirm+val_review):.2f})")
    print(f"Test  - CONFIRM: {test_confirm}, REVIEW: {test_review} (ratio: {test_confirm/(test_confirm+test_review):.2f})")
    print()
    print("Files created:")
    print(f"- {train_file}")
    print(f"- {val_file}")
    print(f"- {test_file}")
    print("\nReady for OpenAI fine-tuning!")

if __name__ == "__main__":
    main()