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
    """Generate 7 typo queries for a student (including sexcode typo)"""
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
            "ground_truth_pen": student["pen_number"],
            "query": query6,
            "original_value": student["mincode"],
            "typo_value": query6["mincode"]
        })
    
    # 7. Sex code: Flip M/F (NEW - moved from conflict)
    if student["sexCode"]:
        query7 = original.copy()
        query7["sexCode"] = flip_sex_code(student["sexCode"])
        queries.append({
            "query_type": "sex_code_typo",
            "ground_truth_pen": student["pen_number"],
            "query": query7,
            "original_value": student["sexCode"],
            "typo_value": query7["sexCode"]
        })
    
    return queries

def generate_typo_dataset():
    """Generate typo queries from the CSV dataset"""
    input_file = "app/evaluation/Test_dataset_100.csv"
    output_file = "app/evaluation/test_queries_typo.json"
    
    all_queries = []
    processed_count = 0
    
    try:
        # Read the CSV file with proper encoding handling
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as file:
            # Read raw lines to handle malformed CSV
            lines = file.readlines()
            
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
                    # Generate typo queries for this student
                    typo_queries = generate_typo_queries(student)
                    all_queries.extend(typo_queries)
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    except FileNotFoundError:
        print(f"File {input_file} not found!")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Save to JSON file
    try:
        output_data = {
            "metadata": {
                "description": "Typo test queries for PEN matching evaluation",
                "purpose": "Test system ability to find correct PEN when input contains typos or data entry errors",
                "total_students_processed": processed_count,
                "total_queries_generated": len(all_queries),
                "query_types": ["first_name_typo", "last_name_typo", "middle_name_typo", 
                              "dob_typo", "postal_code_typo", "mincode_typo", "sex_code_typo"],
                "queries_per_student": 7
            },
            "queries": all_queries
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Generated {len(all_queries)} typo queries from {processed_count} students")
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

if __name__ == "__main__":
    generate_typo_dataset()