import csv
import json
import random
import string
from typing import List, Dict, Any

def change_last_letter(text: str) -> str:
    """Change the last letter of a string to a random letter"""
    if not text or len(text) < 1:
        return text
    
    # Get all letters except the current last letter
    current_last = text[-1].lower()
    available_letters = [c for c in string.ascii_lowercase if c != current_last]
    
    if not available_letters:
        return text
    
    new_last_letter = random.choice(available_letters)
    
    # Preserve case
    if text[-1].isupper():
        new_last_letter = new_last_letter.upper()
    
    return text[:-1] + new_last_letter

def change_last_two_digits(text: str) -> str:
    """Change the last two digits/characters of a string"""
    if not text or len(text) < 2:
        return text
    
    # For numeric strings, change to different digits
    if text[-2:].isdigit():
        current_digits = text[-2:]
        while True:
            new_digits = f"{random.randint(0, 99):02d}"
            if new_digits != current_digits:
                return text[:-2] + new_digits
    else:
        # For alphanumeric, change last two characters
        available_chars = string.ascii_uppercase + string.digits
        new_chars = ""
        for i in range(2):
            pos = len(text) - 2 + i
            if pos < len(text):
                current_char = text[pos]
                available = [c for c in available_chars if c != current_char]
                new_chars += random.choice(available) if available else current_char
        return text[:-2] + new_chars
    
    return text

def flip_sex_code(sex_code: str) -> str:
    """Flip sex code between M and F"""
    if sex_code.upper() == "M":
        return "F"
    elif sex_code.upper() == "F":
        return "M"
    else:
        # If unknown format, randomly assign M or F
        return random.choice(["M", "F"])

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

def generate_typo_queries(student: Dict[str, str]) -> List[Dict[str, Any]]:
    """Generate 7 typo queries for a student with CONFIRM review_label"""
    queries = []
    
    # Original record for reference
    original = {
        "legalFirstName": student["legalFirstName"],
        "legalLastName": student["legalLastName"], 
        "legalMiddleNames": student["legalMiddleNames"],
        "dob": student["dob"],
        "sexCode": student["sexCode"],
        "postalCode": student["postalCode"],
        "mincode": student["mincode"]
    }
    
    # 1. First name: Change the last letter
    if student["legalFirstName"]:
        query1 = original.copy()
        query1["legalFirstName"] = change_last_letter(student["legalFirstName"])
        queries.append({
            "query_type": "first_name_typo",
            "review_label": "CONFIRM",
            "ground_truth_pen": student["pen_number"],
            "query": query1,
            "original_value": student["legalFirstName"],
            "typo_value": query1["legalFirstName"]
        })
    
    # 2. Last name: Change the last letter  
    if student["legalLastName"]:
        query2 = original.copy()
        query2["legalLastName"] = change_last_letter(student["legalLastName"])
        queries.append({
            "query_type": "last_name_typo",
            "review_label": "CONFIRM",
            "ground_truth_pen": student["pen_number"],
            "query": query2,
            "original_value": student["legalLastName"],
            "typo_value": query2["legalLastName"]
        })
    
    # 3. Middle name: Change the last letter
    if student["legalMiddleNames"]:
        query3 = original.copy()
        query3["legalMiddleNames"] = change_last_letter(student["legalMiddleNames"])
        queries.append({
            "query_type": "middle_name_typo",
            "review_label": "CONFIRM",
            "ground_truth_pen": student["pen_number"],
            "query": query3,
            "original_value": student["legalMiddleNames"],
            "typo_value": query3["legalMiddleNames"]
        })
    
    # 4. DOB: Change the last two digits (day part)
    query4 = original.copy()
    query4["dob"] = change_last_two_digits(student["dob"])
    queries.append({
        "query_type": "dob_typo",
        "review_label": "CONFIRM",
        "ground_truth_pen": student["pen_number"],
        "query": query4,
        "original_value": student["dob"],
        "typo_value": query4["dob"]
    })
    
    # 5. Postal code: Change the last two characters
    if student["postalCode"]:
        query5 = original.copy()
        query5["postalCode"] = change_last_two_digits(student["postalCode"])
        queries.append({
            "query_type": "postal_code_typo",
            "review_label": "CONFIRM",
            "ground_truth_pen": student["pen_number"],
            "query": query5,
            "original_value": student["postalCode"],
            "typo_value": query5["postalCode"]
        })
    
    # 6. Mincode: Change the last two digits
    if student["mincode"]:
        query6 = original.copy()
        query6["mincode"] = change_last_two_digits(student["mincode"])
        queries.append({
            "query_type": "mincode_typo",
            "review_label": "CONFIRM",
            "ground_truth_pen": student["pen_number"],
            "query": query6,
            "original_value": student["mincode"],
            "typo_value": query6["mincode"]
        })
    
    # 7. Sex code: Flip M/F
    if student["sexCode"]:
        query7 = original.copy()
        query7["sexCode"] = flip_sex_code(student["sexCode"])
        queries.append({
            "query_type": "sex_code_typo",
            "review_label": "CONFIRM",
            "ground_truth_pen": student["pen_number"],
            "query": query7,
            "original_value": student["sexCode"],
            "typo_value": query7["sexCode"]
        })
    
    return queries

def generate_conflict_queries(target_student: Dict[str, str], all_students: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Generate 3 conflict queries for a student with REVIEW review_label"""
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
    
    # 8. Wrong DOB (use another student's DOB)
    conflict_student_1 = random.choice(other_students)
    query8 = base_query.copy()
    query8["dob"] = conflict_student_1["dob"]
    queries.append({
        "query_type": "wrong_dob",
        "review_label": "REVIEW",
        "ground_truth_pen": target_student["pen_number"],
        "conflict_source_pen": conflict_student_1["pen_number"],
        "query": query8,
        "original_value": target_student["dob"],
        "conflict_value": conflict_student_1["dob"]
    })
    
    # 9. Wrong postal code (use another student's postal code)
    conflict_student_2 = random.choice([s for s in other_students if s["pen_number"] != conflict_student_1["pen_number"]])
    query9 = base_query.copy()
    query9["postalCode"] = conflict_student_2["postalCode"]
    queries.append({
        "query_type": "wrong_postal_code",
        "review_label": "REVIEW",
        "ground_truth_pen": target_student["pen_number"],
        "conflict_source_pen": conflict_student_2["pen_number"],
        "query": query9,
        "original_value": target_student["postalCode"],
        "conflict_value": conflict_student_2["postalCode"]
    })
    
    # 10. Wrong mincode (use another student's mincode)
    conflict_student_3 = random.choice([s for s in other_students 
                                      if s["pen_number"] not in [conflict_student_1["pen_number"], 
                                                                conflict_student_2["pen_number"]]])
    query10 = base_query.copy()
    query10["mincode"] = conflict_student_3["mincode"]
    queries.append({
        "query_type": "wrong_mincode",
        "review_label": "REVIEW",
        "ground_truth_pen": target_student["pen_number"],
        "conflict_source_pen": conflict_student_3["pen_number"],
        "query": query10,
        "original_value": target_student["mincode"],
        "conflict_value": conflict_student_3["mincode"]
    })
    
    return queries

def generate_review_dataset():
    """Generate comprehensive review queries from the CSV dataset"""
    input_file = "app/evaluation/Test_dataset_100.csv"
    output_file = "app/evaluation/test_queries_review.json"
    
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
    
    # Second pass: generate all queries for each student
    processed_count = 0
    for student in all_students:
        try:
            # Generate typo queries (7 queries with CONFIRM review_label)
            typo_queries = generate_typo_queries(student)
            all_queries.extend(typo_queries)
            
            # Generate conflict queries (3 queries with REVIEW review_label)
            conflict_queries = generate_conflict_queries(student, all_students)
            all_queries.extend(conflict_queries)
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} students...")
                
        except Exception as e:
            print(f"Error generating queries for PEN {student['pen_number']}: {e}")
            continue
    
    # Save to JSON file
    try:
        # Count queries by type and review_label
        query_type_counts = {}
        review_label_counts = {"CONFIRM": 0, "REVIEW": 0}
        
        for query in all_queries:
            qtype = query["query_type"]
            review_label = query["review_label"]
            query_type_counts[qtype] = query_type_counts.get(qtype, 0) + 1
            review_label_counts[review_label] += 1
        
        output_data = {
            "metadata": {
                "description": "Comprehensive review test queries for PEN matching evaluation",
                "purpose": "Test system ability to find correct PEN with both typos (CONFIRM) and conflicts (REVIEW)",
                "total_students_processed": processed_count,
                "total_queries_generated": len(all_queries),
                "queries_per_student": 10,
                "query_categories": {
                    "typo_queries": {
                        "review_label": "CONFIRM",
                        "count": review_label_counts["CONFIRM"],
                        "types": ["first_name_typo", "last_name_typo", "middle_name_typo", 
                                "dob_typo", "postal_code_typo", "mincode_typo", "sex_code_typo"]
                    },
                    "conflict_queries": {
                        "review_label": "REVIEW", 
                        "count": review_label_counts["REVIEW"],
                        "types": ["wrong_dob", "wrong_postal_code", "wrong_mincode"]
                    }
                },
                "query_type_breakdown": query_type_counts
            },
            "queries": all_queries
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nGenerated {len(all_queries)} total queries from {processed_count} students")
        print(f"Output saved to {output_file}")
        
        # Print detailed summary
        print(f"\nQuery breakdown:")
        print(f"  CONFIRM queries (typos): {review_label_counts['CONFIRM']}")
        print(f"  REVIEW queries (conflicts): {review_label_counts['REVIEW']}")
        print(f"  Total: {len(all_queries)}")
        
        print(f"\nDetailed query type counts:")
        for qtype, count in sorted(query_type_counts.items()):
            review_label = "CONFIRM" if "typo" in qtype else "REVIEW"
            print(f"  {qtype} ({review_label}): {count}")
        
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    generate_review_dataset()