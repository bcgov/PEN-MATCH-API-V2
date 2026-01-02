import csv
import json
import random
from typing import List, Dict, Any

def parse_student_record(row: List[str]) -> Dict[str, str]:
    """Parse a CSV row into student record"""
    if len(row) < 23:
        return None
    
    # Extract student data
    pen_number = row[1].strip()
    first_name = row[2].strip()
    middle_name = row[3].strip() if len(row) > 3 else ""
    last_name = row[4].strip()
    sex_code = row[6].strip() if len(row) > 6 else ""
    postal_code = row[18].strip() if len(row) > 18 else ""
    mincode = row[21].strip() if len(row) > 21 else ""
    
    # Generate a realistic DOB (format: YYYY-MM-DD)
    year = random.randint(1990, 2010)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # Use 28 to avoid month-specific day issues
    dob = f"{year}-{month:02d}-{day:02d}"
    
    return {
        "pen_number": pen_number,
        "legalFirstName": first_name,
        "legalMiddleNames": middle_name,
        "legalLastName": last_name,
        "dob": dob,
        "sexCode": sex_code,
        "postalCode": postal_code,
        "mincode": mincode
    }

def generate_conflict_queries(target_student: Dict[str, str], all_students: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Generate 3 conflict queries for a student using other students' data (removed sexcode)"""
    queries = []
    
    # Get other students for conflict data
    other_students = [s for s in all_students if s["pen_number"] != target_student["pen_number"]]
    
    if len(other_students) < 3:
        print(f"Warning: Not enough other students for conflicts for PEN {target_student['pen_number']}")
        return queries
    
    # Base query with correct name information
    base_query = {
        "legalFirstName": target_student["legalFirstName"],
        "legalLastName": target_student["legalLastName"], 
        "legalMiddleNames": target_student["legalMiddleNames"],
        "dob": target_student["dob"],
        "sexCode": target_student["sexCode"],
        "postalCode": target_student["postalCode"],
        "mincode": target_student["mincode"]
    }
    
    # 1. Wrong DOB (use another student's DOB)
    conflict_student_1 = random.choice(other_students)
    query1 = base_query.copy()
    query1["dob"] = conflict_student_1["dob"]
    queries.append({
        "query_type": "wrong_dob",
        "ground_truth_pen": target_student["pen_number"],
        "conflict_source_pen": conflict_student_1["pen_number"],
        "query": query1,
        "original_value": target_student["dob"],
        "conflict_value": conflict_student_1["dob"]
    })
    
    # 2. Wrong postal code (use another student's postal code)
    conflict_student_2 = random.choice([s for s in other_students if s["pen_number"] != conflict_student_1["pen_number"]])
    query2 = base_query.copy()
    query2["postalCode"] = conflict_student_2["postalCode"]
    queries.append({
        "query_type": "wrong_postal_code",
        "ground_truth_pen": target_student["pen_number"],
        "conflict_source_pen": conflict_student_2["pen_number"],
        "query": query2,
        "original_value": target_student["postalCode"],
        "conflict_value": conflict_student_2["postalCode"]
    })
    
    # 3. Wrong mincode (use another student's mincode)
    conflict_student_3 = random.choice([s for s in other_students 
                                      if s["pen_number"] not in [conflict_student_1["pen_number"], 
                                                                conflict_student_2["pen_number"]]])
    query3 = base_query.copy()
    query3["mincode"] = conflict_student_3["mincode"]
    queries.append({
        "query_type": "wrong_mincode",
        "ground_truth_pen": target_student["pen_number"],
        "conflict_source_pen": conflict_student_3["pen_number"],
        "query": query3,
        "original_value": target_student["mincode"],
        "conflict_value": conflict_student_3["mincode"]
    })
    
    # NOTE: Removed sex_code conflict - moved to typo queries
    
    return queries

def generate_conflict_dataset():
    """Generate conflict queries from the CSV dataset"""
    input_file = "app/evaluation/Test_dataset_100.csv"
    output_file = "app/evaluation/test_queries_conflict_v2.json"
    
    all_students = []
    all_queries = []
    
    try:
        # Read the CSV file with proper encoding handling
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
            
        # First pass: collect all student records
        for line_num, line in enumerate(lines, 1):
            try:
                # Split by comma, handling quoted fields
                row = []
                current_field = ""
                in_quotes = False
                
                i = 0
                while i < len(line):
                    char = line[i]
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        row.append(current_field)
                        current_field = ""
                    else:
                        current_field += char
                    i += 1
                
                # Add the last field
                if current_field:
                    row.append(current_field.strip())
                
                # Parse student record
                student = parse_student_record(row)
                if student and student["pen_number"]:
                    all_students.append(student)
                    
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    except FileNotFoundError:
        print(f"File {input_file} not found!")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print(f"Collected {len(all_students)} students")
    
    # Second pass: generate conflict queries for each student
    processed_count = 0
    for student in all_students:
        try:
            conflict_queries = generate_conflict_queries(student, all_students)
            all_queries.extend(conflict_queries)
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} students...")
                
        except Exception as e:
            print(f"Error generating conflicts for PEN {student['pen_number']}: {e}")
            continue
    
    # Save to JSON file
    try:
        output_data = {
            "metadata": {
                "description": "Conflict test queries for PEN matching evaluation",
                "purpose": "Test system ability to find correct PEN when some demographic data conflicts with other students",
                "total_students_processed": processed_count,
                "total_queries_generated": len(all_queries),
                "query_types": ["wrong_dob", "wrong_postal_code", "wrong_mincode"],
                "queries_per_student": 3,
                "note": "Sex code conflicts moved to typo queries as they represent data entry errors rather than true conflicts"
            },
            "queries": all_queries
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nGenerated {len(all_queries)} conflict queries from {processed_count} students")
        print(f"Output saved to {output_file}")
        
        # Print summary by query type
        query_types = {}
        for query in all_queries:
            qtype = query["query_type"]
            query_types[qtype] = query_types.get(qtype, 0) + 1
        
        print("\nQuery type breakdown:")
        for qtype, count in query_types.items():
            print(f"  {qtype}: {count}")
        
    except Exception as e:
        print(f"Error saving output file: {e}")

def analyze_conflict_potential():
    """Analyze the dataset to understand conflict potential"""
    input_file = "app/evaluation/Test_dataset_100.csv"
    
    all_students = []
    
    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
            
        for line_num, line in enumerate(lines, 1):
            try:
                row = []
                current_field = ""
                in_quotes = False
                
                i = 0
                while i < len(line):
                    char = line[i]
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        row.append(current_field)
                        current_field = ""
                    else:
                        current_field += char
                    i += 1
                
                if current_field:
                    row.append(current_field.strip())
                
                student = parse_student_record(row)
                if student and student["pen_number"]:
                    all_students.append(student)
                    
            except Exception as e:
                continue
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Analyze diversity for conflicts
    sex_codes = set(s["sexCode"] for s in all_students if s["sexCode"])
    postal_codes = set(s["postalCode"] for s in all_students if s["postalCode"])
    mincodes = set(s["mincode"] for s in all_students if s["mincode"])
    
    print(f"Dataset analysis:")
    print(f"  Total students: {len(all_students)}")
    print(f"  Unique sex codes: {len(sex_codes)} - {sorted(sex_codes)}")
    print(f"  Unique postal codes: {len(postal_codes)}")
    print(f"  Unique mincodes: {len(mincodes)}")
    print(f"  Conflict potential: High" if len(postal_codes) > 10 and len(mincodes) > 10 else "Conflict potential: Medium")

if __name__ == "__main__":
    print("Analyzing dataset for conflict potential...")
    analyze_conflict_potential()
    print("\nGenerating conflict queries...")
    generate_conflict_dataset()