import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.student_match import StudentWorkflow

def test_student_match_workflow():
    """Test the complete student matching workflow"""
    try:
        print("=" * 60)
        print("TESTING STUDENT MATCH WORKFLOW")
        print("=" * 60)
        
        # Initialize workflow
        workflow = StudentWorkflow()
        
        # Step 1: Bulk import 100 students into Cosmos
        print("\n1. BULK IMPORTING STUDENTS INTO COSMOS DB")
        print("-" * 40)
        imported_count = workflow.bulk_import_students(page_size=100, max_pages=1)
        print(f"Successfully imported {imported_count} students")
        
        # Step 2: Get some existing students from source to test with
        print("\n2. FETCHING EXISTING STUDENTS FOR TESTING")
        print("-" * 40)
        existing_students = workflow.student_api.get_student_page(page=1, size=5)
        
        if not existing_students:
            print("No students found in source database")
            return
            
        print(f"Retrieved {len(existing_students)} students for testing")
        workflow.student_api.print_student_info(existing_students[:2])
        
        # Step 3: Test with existing student (should find perfect match)
        print("\n3. TESTING WITH EXISTING STUDENT")
        print("-" * 40)
        test_student = existing_students[0]
        print(f"Testing with: {test_student.get('legalFirstName')} {test_student.get('legalLastName')}")
        
        result = workflow.process_student_query(test_student)
        print(f"Result status: {result['status']}")
        
        if result['status'] == 'perfect_match_found':
            print(f"✅ Perfect match found!")
            print(f"   Similarity score: {result['similarity_score']:.4f}")
            print(f"   Source: {result['source']}")
        elif result['status'] == 'no_perfect_match':
            print(f"⚠️ No perfect match found")
            print(f"   Best score: {result['best_score']:.4f}")
            print(f"   Found {len(result['candidates'])} candidates")
        else:
            print(f"❌ Unexpected result: {result}")
        
        # Step 4: Get students from a different page for testing new student scenario
        print("\n4. FETCHING STUDENTS FROM DIFFERENT PAGE FOR NEW STUDENT TEST")
        print("-" * 40)
        new_page_students = workflow.student_api.get_student_page(page=2, size=10)
        
        if not new_page_students:
            print("No students found on page 2")
            return
        
        # Find a student that doesn't exist in Cosmos yet
        new_test_student = None
        for student in new_page_students:
            first_name = student.get('legalFirstName')
            last_name = student.get('legalLastName')
            
            if first_name and last_name:
                # Check if this name exists in Cosmos
                if not workflow.cosmos_client.name_exists(first_name, last_name):
                    new_test_student = student
                    break
        
        if not new_test_student:
            print("All students from page 2 already exist in Cosmos, using first student anyway")
            new_test_student = new_page_students[0]
        
        print(f"Selected student for new test: {new_test_student.get('legalFirstName')} {new_test_student.get('legalLastName')}")
        workflow.student_api.print_student_info([new_test_student])
        
        # Step 5: Test with the new student (should not exist in Cosmos initially)
        print("\n5. TESTING WITH NEW STUDENT FROM DIFFERENT PAGE")
        print("-" * 40)
        
        # Check if this name exists in Cosmos before test
        name_exists_before = workflow.cosmos_client.name_exists(
            new_test_student['legalFirstName'], 
            new_test_student['legalLastName']
        )
        print(f"Name exists in Cosmos before test: {name_exists_before}")
        
        # Process the query
        result = workflow.process_student_query(new_test_student)
        print(f"Result status: {result['status']}")
        
        if result['status'] == 'perfect_match_found':
            print(f"✅ Perfect match found!")
            print(f"   Similarity score: {result['similarity_score']:.4f}")
            print(f"   Source: {result['source']}")
        elif result['status'] == 'no_perfect_match':
            print(f"⚠️ No perfect match found")
            print(f"   Best score: {result['best_score']:.4f}")
            print(f"   Found {len(result['candidates'])} candidates")
        elif result['status'] == 'no_students_found':
            print(f"ℹ️ No students found with that name in source database")
        else:
            print(f"Result: {result}")
        
        # Check if name now exists in Cosmos
        name_exists_after = workflow.cosmos_client.name_exists(
            new_test_student['legalFirstName'], 
            new_test_student['legalLastName']
        )
        print(f"Name exists in Cosmos after test: {name_exists_after}")
        
        # Step 6: Verify the student data in Cosmos
        print("\n6. VERIFYING STUDENT DATA IN COSMOS DB")
        print("-" * 40)
        cosmos_students = workflow.cosmos_client.get_students_by_name(
            new_test_student['legalFirstName'], 
            new_test_student['legalLastName']
        )
        print(f"Found {len(cosmos_students)} students with name {new_test_student['legalFirstName']} {new_test_student['legalLastName']} in Cosmos")
        
        if cosmos_students:
            for i, student in enumerate(cosmos_students, 1):
                print(f"   Student {i}:")
                print(f"      PEN: {student.get('pen')}")
                print(f"      Name: {student.get('legalFirstName')} {student.get('legalMiddleNames', '')} {student.get('legalLastName')}")
                print(f"      DOB: {student.get('dob')}")
                print(f"      Local ID: {student.get('localID')}")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_student_match_workflow()