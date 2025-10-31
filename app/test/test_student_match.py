import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.student_match import StudentWorkflow

def clear_cosmos_database():
    """Clear all content from Cosmos database"""
    print("Clearing Cosmos database...")
    workflow = StudentWorkflow()
    
    # Get all students
    query = "SELECT * FROM c"
    all_students = list(workflow.cosmos_client.container.query_items(
        query=query,
        enable_cross_partition_query=True
    ))
    
    print(f"Found {len(all_students)} students to delete")
    
    # Delete all students
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
    
    print(f"Deleted {deleted_count} students from Cosmos")
    return deleted_count

def test_existing_student_match():
    """Test 1: Import first student's name and test embedding match"""
    print("TEST 1: EXISTING STUDENT EMBEDDING MATCH")
    print("=" * 50)
    
    workflow = StudentWorkflow()
    
    # Get first student from source
    students = workflow.student_api.get_student_page(page=1, size=1)
    if not students:
        print("No students found")
        return False
    
    first_student = students[0]
    first_name = first_student.get('legalFirstName')
    last_name = first_student.get('legalLastName')
    
    print(f"Selected: {first_name} {last_name} (PEN: {first_student.get('pen')})")
    
    # Get ALL students with this name and import to Cosmos
    all_same_name = workflow.student_api.get_students_by_name(first_name, last_name)
    print(f"Found {len(all_same_name)} students with this name in source")
    
    workflow.create_embeddings_for_students(all_same_name)
    
    # Verify import - should match the number from source
    cosmos_students = workflow.cosmos_client.get_students_by_name(first_name, last_name)
    print(f"Imported {len(cosmos_students)} students to Cosmos")
    
    # Test embedding match
    result = workflow.process_student_query(first_student)
    
    if result['status'] == 'perfect_match_found':
        print(f"Perfect match found (score: {result['similarity_score']:.4f})")
        return True
    else:
        print(f"No perfect match (best score: {result.get('best_score', 0):.4f})")
        return False

def test_new_student_import():
    """Test 2: Import student whose name doesn't exist in Cosmos"""
    print("\nTEST 2: NEW STUDENT NAME IMPORT AND MATCH")
    print("=" * 50)
    
    workflow = StudentWorkflow()
    
    # Find student that doesn't exist in Cosmos by checking more pages
    test_student = None
    pages_checked = 0
    
    for page in range(2, 21):  # Check pages 2-20
        students = workflow.student_api.get_student_page(page=page, size=10)
        pages_checked += 1
        
        if not students:
            continue
            
        for student in students:
            first_name = student.get('legalFirstName')
            last_name = student.get('legalLastName')
            
            if first_name and last_name:
                exists = workflow.cosmos_client.name_exists(first_name, last_name)
                if not exists:
                    test_student = student
                    print(f"Found non-existing student on page {page}: {first_name} {last_name}")
                    break
        
        if test_student:
            break
    
    if not test_student:
        print(f"Could not find non-existing student after checking {pages_checked} pages")
        print("All students seem to already exist in Cosmos")
        return False
    
    student_name = test_student.get('legalFirstName')
    student_last = test_student.get('legalLastName')
    
    print(f"Selected: {student_name} {student_last} (PEN: {test_student.get('pen')})")
    
    # Verify name doesn't exist before test
    name_exists_before = workflow.cosmos_client.name_exists(student_name, student_last)
    print(f"Exists in Cosmos before: {name_exists_before}")
    
    if name_exists_before:
        print("Error: Selected student already exists in Cosmos!")
        return False
    
    # Get ALL students with this name from source
    source_students = workflow.student_api.get_students_by_name(student_name, student_last)
    print(f"Found {len(source_students)} students with this name in source")
    
    if len(source_students) == 0:
        print("No students found in source")
        return False
    
    # Process query (should import ALL students with this name)
    result = workflow.process_student_query(test_student)
    
    # Verify import happened
    name_exists_after = workflow.cosmos_client.name_exists(student_name, student_last)
    cosmos_students = workflow.cosmos_client.get_students_by_name(student_name, student_last)
    
    print(f"Exists in Cosmos after: {name_exists_after}")
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
    print("=" * 80)
    
    try:
        # Clear database before starting tests
        clear_cosmos_database()
        print()
        
        test1_success = test_existing_student_match()
        
        # Clear database before test 2 to ensure clean state
        print(f"\nClearing database before Test 2...")
        clear_cosmos_database()
        print()
        
        test2_success = test_new_student_import()
        
        print(f"\nRESULTS:")
        print(f"Test 1: {'PASSED' if test1_success else 'FAILED'}")
        print(f"Test 2: {'PASSED' if test2_success else 'FAILED'}")
        
        if test1_success and test2_success:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("Some tests failed")
            
    except Exception as e:
        print(f"Tests failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()