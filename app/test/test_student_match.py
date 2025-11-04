import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.student_match import StudentWorkflow

def clear_cosmos_database():
    """Clear all content from Cosmos database"""
    print("Clearing Cosmos database...")
    workflow = StudentWorkflow()
    
    query = "SELECT * FROM c"
    all_students = list(workflow.cosmos_client.container.query_items(
        query=query,
        enable_cross_partition_query=True
    ))
    
    deleted_count = 0
    for student in all_students:
        try:
            workflow.cosmos_client.container.delete_item(
                item=student['id'], 
                partition_key=student['pen']
            )
            deleted_count += 1
        except Exception as e:
            print(f"Failed to delete student {student.get('pen')}: {str(e)}")
    
    print(f"Deleted {deleted_count} students")
    return deleted_count

def test_existing_student_match():
    """Test 1: Import first student's name and test embedding match"""
    print("TEST 1: EXISTING STUDENT EMBEDDING MATCH")
    
    workflow = StudentWorkflow()
    
    students = workflow.student_api.get_student_page(page=1, size=1)
    if not students:
        print("No students found")
        return False
    
    first_student = students[0]
    first_name = first_student.get('legalFirstName')
    last_name = first_student.get('legalLastName')
    
    print(f"Selected: {first_name} {last_name} (PEN: {first_student.get('pen')})")
    
    all_same_name = workflow.student_api.get_students_by_name(first_name, last_name)
    print(f"Found {len(all_same_name)} students in source")
    
    workflow.create_embeddings_for_students(all_same_name)
    
    cosmos_students = workflow.cosmos_client.get_students_by_name(first_name, last_name)
    print(f"Imported {len(cosmos_students)} students to Cosmos")
    
    result = workflow.process_student_query(first_student)
    
    if result['status'] == 'perfect_match_found':
        print(f"Perfect match found (score: {result['similarity_score']:.4f})")
        return True
    else:
        print(f"No perfect match (best score: {result.get('best_score', 0):.4f})")
        return False

def test_new_student_import():
    """Test 2: Import student from page 2"""
    print("TEST 2: NEW STUDENT NAME IMPORT AND MATCH")
    
    workflow = StudentWorkflow()
    
    students = workflow.student_api.get_student_page(page=2, size=1)
    if not students:
        print("No students found on page 2")
        return False
    
    test_student = students[0]
    student_name = test_student.get('legalFirstName')
    student_last = test_student.get('legalLastName')
    
    print(f"Selected: {student_name} {student_last} (PEN: {test_student.get('pen')})")
    
    source_students = workflow.student_api.get_students_by_name(student_name, student_last)
    print(f"Found {len(source_students)} students in source")
    
    if len(source_students) == 0:
        print("No students found in source")
        return False
    
    result = workflow.process_student_query(test_student)
    
    cosmos_students = workflow.cosmos_client.get_students_by_name(student_name, student_last)
    print(f"Students in Cosmos: {len(cosmos_students)} (expected: {len(source_students)})")

    if result['status'] == 'perfect_match_found':
        print(f"Perfect match found (score: {result['similarity_score']:.4f})")
        return True
    else:
        print(f"No perfect match (best score: {result.get('best_score', 0):.4f})")
        return False

def run_all_tests():
    """Run both tests with database clearing"""
    print("STUDENT MATCH TESTS")
    
    try:
        clear_cosmos_database()
        
        test1_success = test_existing_student_match()
        
        clear_cosmos_database()
        
        test2_success = test_new_student_import()
        
        print(f"Test 1: {'PASSED' if test1_success else 'FAILED'}")
        print(f"Test 2: {'PASSED' if test2_success else 'FAILED'}")
        
        if test1_success and test2_success:
            print("ALL TESTS PASSED")
        else:
            print("Some tests failed")
            
    except Exception as e:
        print(f"Tests failed: {str(e)}")

if __name__ == "__main__":
    run_all_tests()