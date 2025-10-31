import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.student_match import StudentWorkflow

def test_existing_student_match():
    """Test 1: Import first student's name and test embedding match"""
    print("=" * 60)
    print("TEST 1: EXISTING STUDENT EMBEDDING MATCH")
    print("=" * 60)
    
    workflow = StudentWorkflow()
    
    # Get first student from source
    students = workflow.student_api.get_student_page(page=1, size=1)
    if not students:
        print("❌ No students found in source database")
        return False
    
    first_student = students[0]
    first_name = first_student.get('legalFirstName')
    last_name = first_student.get('legalLastName')
    
    print(f"Selected student: {first_name} {last_name}")
    workflow.student_api.print_student_info([first_student])
    
    # Get all students with this name from source
    all_same_name = workflow.student_api.get_students_by_name(first_name, last_name)
    print(f"\nFound {len(all_same_name)} students with name '{first_name} {last_name}' in source")
    
    # Import all students with this name to Cosmos
    print(f"\nImporting {len(all_same_name)} students to Cosmos...")
    workflow.create_embeddings_for_students(all_same_name)
    
    # Test embedding match
    print(f"\nTesting embedding match for: {first_name} {last_name}")
    result = workflow.process_student_query(first_student)
    
    print(f"Result status: {result['status']}")
    if result['status'] == 'perfect_match_found':
        print(f"✅ Perfect match found!")
        print(f"   Similarity score: {result['similarity_score']:.4f}")
        print(f"   Matched PEN: {result['student'].get('pen')}")
        print(f"   Original PEN: {first_student.get('pen')}")
        print(f"   PENs match: {result['student'].get('pen') == first_student.get('pen')}")
        return True
    elif result['status'] == 'no_perfect_match':
        print(f"⚠️ No perfect match found")
        print(f"   Best score: {result['best_score']:.4f}")
        print(f"   Found {len(result['candidates'])} candidates")
        return False
    else:
        print(f"❌ Unexpected result: {result}")
        return False

def test_new_student_import():
    """Test 2: Import second student whose name doesn't exist in Cosmos"""
    print("\n" + "=" * 60)
    print("TEST 2: NEW STUDENT NAME IMPORT AND MATCH")
    print("=" * 60)
    
    workflow = StudentWorkflow()
    
    # Get second student from source
    students = workflow.student_api.get_student_page(page=1, size=2)
    if len(students) < 2:
        print("❌ Need at least 2 students for this test")
        return False
    
    second_student = students[1]
    second_name = second_student.get('legalFirstName')
    second_last = second_student.get('legalLastName')
    
    print(f"Selected second student: {second_name} {second_last}")
    workflow.student_api.print_student_info([second_student])
    
    # Check if this name exists in Cosmos before test
    name_exists_before = workflow.cosmos_client.name_exists(second_name, second_last)
    print(f"\nName exists in Cosmos before test: {name_exists_before}")
    
    if name_exists_before:
        print("⚠️ Name already exists in Cosmos, but continuing test...")
    
    # Process query (should trigger import if name doesn't exist)
    print(f"\nProcessing query for: {second_name} {second_last}")
    result = workflow.process_student_query(second_student)
    
    print(f"Result status: {result['status']}")
    if result['status'] == 'perfect_match_found':
        print(f"✅ Perfect match found!")
        print(f"   Similarity score: {result['similarity_score']:.4f}")
        print(f"   Source: {result['source']}")
        print(f"   Matched PEN: {result['student'].get('pen')}")
        print(f"   Original PEN: {second_student.get('pen')}")
        print(f"   PENs match: {result['student'].get('pen') == second_student.get('pen')}")
        success = True
    elif result['status'] == 'no_perfect_match':
        print(f"⚠️ No perfect match found")
        print(f"   Best score: {result['best_score']:.4f}")
        print(f"   Found {len(result['candidates'])} candidates")
        print(f"   Source: {result['source']}")
        success = False
    elif result['status'] == 'no_students_found':
        print(f"ℹ️ No students found with name: {second_name} {second_last}")
        success = False
    else:
        print(f"❌ Unexpected result: {result}")
        success = False
    
    # Verify name now exists in Cosmos
    name_exists_after = workflow.cosmos_client.name_exists(second_name, second_last)
    print(f"\nName exists in Cosmos after test: {name_exists_after}")
    
    if name_exists_after and not name_exists_before:
        print("✅ Successfully imported new name to Cosmos")
    elif name_exists_after:
        print("ℹ️ Name confirmed to exist in Cosmos")
    else:
        print("❌ Name still doesn't exist in Cosmos")
    
    # Show all students with this name in Cosmos
    cosmos_students = workflow.cosmos_client.get_students_by_name(second_name, second_last)
    print(f"\nStudents in Cosmos with name '{second_name} {second_last}': {len(cosmos_students)}")
    for i, student in enumerate(cosmos_students, 1):
        print(f"   {i}. PEN: {student.get('pen')}, DOB: {student.get('dob')}")
    
    return success

def run_all_tests():
    """Run both tests"""
    print("STARTING STUDENT MATCH TESTS")
    print("=" * 80)
    
    try:
        # Test 1: Existing student embedding match
        test1_success = test_existing_student_match()
        
        # Test 2: New student import and match
        test2_success = test_new_student_import()
        
        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Test 1 (Existing Student Match): {'✅ PASSED' if test1_success else '❌ FAILED'}")
        print(f"Test 2 (New Student Import): {'✅ PASSED' if test2_success else '❌ FAILED'}")
        
        if test1_success and test2_success:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n⚠️ Some tests failed. Please review the results above.")
            
    except Exception as e:
        print(f"\n❌ Tests failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()