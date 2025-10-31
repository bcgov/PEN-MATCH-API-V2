import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.cosmos_client import CosmosDBClient
from app.core.student_embedding import StudentEmbedding

class CosmosTestSuite:
    def __init__(self):
        self.cosmos_client = CosmosDBClient()
        self.embedding_service = StudentEmbedding()

    def get_all_students_paginated(self, page_size=20, max_pages=None):
        """Get all students with pagination"""
        print(f"Fetching students with page size: {page_size}")
        
        query = "SELECT * FROM c ORDER BY c.pen"
        
        # Get query iterator
        query_iterator = self.cosmos_client.container.query_items(
            query=query,
            enable_cross_partition_query=True,
            max_item_count=page_size
        )
        
        all_students = []
        page_count = 0
        
        # Iterate through pages
        for page in query_iterator.by_page():
            page_count += 1
            page_items = list(page)
            
            print(f"Page {page_count}: {len(page_items)} students")
            all_students.extend(page_items)
            
            # Show first student in each page
            if page_items:
                first_student = page_items[0]
                print(f"  First student: {first_student.get('legalFirstName')} {first_student.get('legalLastName')} (PEN: {first_student.get('pen')})")
            
            # Stop if max pages reached
            if max_pages and page_count >= max_pages:
                print(f"Stopped at page {page_count} (max_pages limit)")
                break
        
        print(f"Total students retrieved: {len(all_students)}")
        return all_students

    def search_students_by_name_pattern(self, name_pattern):
        """Search students by name pattern (contains)"""
        print(f"Searching for students with name pattern: '{name_pattern}'")
        
        query = """
        SELECT * FROM c 
        WHERE CONTAINS(UPPER(c.legalFirstName), UPPER(@pattern)) 
        OR CONTAINS(UPPER(c.legalLastName), UPPER(@pattern))
        """
        parameters = [
            {"name": "@pattern", "value": name_pattern.upper()}
        ]
        
        students = list(self.cosmos_client.container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        print(f"Found {len(students)} students matching pattern '{name_pattern}'")
        return students

    def get_students_by_exact_name(self, first_name, last_name):
        """Get students by exact first and last name"""
        print(f"Searching for exact name: {first_name} {last_name}")
        return self.cosmos_client.get_students_by_name(first_name, last_name)

    def get_student_by_pen(self, pen):
        """Get specific student by PEN"""
        print(f"Searching for student with PEN: {pen}")
        return self.cosmos_client.get_student_by_pen(pen)

    def get_students_by_pen_list(self, pen_list):
        """Get multiple students by list of PENs"""
        print(f"Searching for {len(pen_list)} students by PEN list")
        
        # Build query with IN clause
        pen_params = []
        param_names = []
        for i, pen in enumerate(pen_list):
            param_name = f"@pen{i}"
            pen_params.append({"name": param_name, "value": pen})
            param_names.append(param_name)
        
        query = f"SELECT * FROM c WHERE c.pen IN ({', '.join(param_names)})"
        
        students = list(self.cosmos_client.container.query_items(
            query=query,
            parameters=pen_params,
            enable_cross_partition_query=True
        ))
        
        print(f"Found {len(students)} students from PEN list")
        return students

    def get_students_by_dob_range(self, start_date, end_date):
        """Get students by date of birth range"""
        print(f"Searching for students born between {start_date} and {end_date}")
        
        query = """
        SELECT * FROM c 
        WHERE c.dob >= @start_date 
        AND c.dob <= @end_date
        ORDER BY c.dob
        """
        parameters = [
            {"name": "@start_date", "value": start_date},
            {"name": "@end_date", "value": end_date}
        ]
        
        students = list(self.cosmos_client.container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        print(f"Found {len(students)} students born between {start_date} and {end_date}")
        return students

    def get_database_statistics(self):
        """Get statistics about the database"""
        print("Getting database statistics...")
        
        # Total count
        count_query = "SELECT VALUE COUNT(1) FROM c"
        total_count = list(self.cosmos_client.container.query_items(
            query=count_query,
            enable_cross_partition_query=True
        ))[0]
        
        # Unique names count
        unique_names_query = """
        SELECT VALUE COUNT(1) FROM (
            SELECT DISTINCT c.legalFirstName, c.legalLastName FROM c
        )
        """
        unique_names = list(self.cosmos_client.container.query_items(
            query=unique_names_query,
            enable_cross_partition_query=True
        ))[0]
        
        print(f"Total students: {total_count}")
        print(f"Unique name combinations: {unique_names}")
        
        return {
            "total_students": total_count,
            "unique_names": unique_names
        }

    def print_student_summary(self, students, max_display=5):
        """Print summary of students list"""
        if not students:
            print("No students to display")
            return
        
        print(f"\nDisplaying {min(len(students), max_display)} of {len(students)} students:")
        print("-" * 80)
        
        for i, student in enumerate(students[:max_display]):
            print(f"{i+1}. PEN: {student.get('pen')}")
            print(f"   Name: {student.get('legalFirstName')} {student.get('legalMiddleNames', '')} {student.get('legalLastName')}")
            print(f"   DOB: {student.get('dob')}")
            print(f"   Local ID: {student.get('localID')}")
            print(f"   Has Embedding: {'Yes' if student.get('embedding') else 'No'}")
            if student.get('embedding'):
                print(f"   Embedding Dimensions: {len(student['embedding'])}")
            print()

def test_cosmos_access():
    """Test different ways to access Cosmos DB"""
    try:
        print("=" * 80)
        print("TESTING COSMOS DB ACCESS METHODS")
        print("=" * 80)
        
        test_suite = CosmosTestSuite()
        
        # Test 1: Get database statistics
        print("\n1. DATABASE STATISTICS")
        print("-" * 40)
        stats = test_suite.get_database_statistics()
        
        # Test 2: Get students by pages
        print("\n2. PAGINATED ACCESS")
        print("-" * 40)
        paginated_students = test_suite.get_all_students_paginated(page_size=5, max_pages=2)
        test_suite.print_student_summary(paginated_students)
        
        # Test 3: Search by name pattern
        print("\n3. SEARCH BY NAME PATTERN")
        print("-" * 40)
        pattern_students = test_suite.search_students_by_name_pattern("John")
        test_suite.print_student_summary(pattern_students)
        
        # Test 4: Get specific student by PEN (if we have students)
        if paginated_students:
            print("\n4. GET BY SPECIFIC PEN")
            print("-" * 40)
            test_pen = paginated_students[0].get('pen')
            pen_student = test_suite.get_student_by_pen(test_pen)
            if pen_student:
                test_suite.print_student_summary([pen_student])
            
            # Test 5: Get multiple students by PEN list
            print("\n5. GET BY PEN LIST")
            print("-" * 40)
            pen_list = [s.get('pen') for s in paginated_students[:3] if s.get('pen')]
            if pen_list:
                pen_list_students = test_suite.get_students_by_pen_list(pen_list)
                test_suite.print_student_summary(pen_list_students)
        
        # Test 6: Search by exact name (if we have students)
        if paginated_students:
            print("\n6. SEARCH BY EXACT NAME")
            print("-" * 40)
            test_student = paginated_students[0]
            first_name = test_student.get('legalFirstName')
            last_name = test_student.get('legalLastName')
            
            if first_name and last_name:
                exact_name_students = test_suite.get_students_by_exact_name(first_name, last_name)
                test_suite.print_student_summary(exact_name_students)
        
        # Test 7: Search by date of birth range
        print("\n7. SEARCH BY DATE OF BIRTH RANGE")
        print("-" * 40)
        dob_students = test_suite.get_students_by_dob_range("2000-01-01", "2010-12-31")
        test_suite.print_student_summary(dob_students)
        
        print("\n" + "=" * 80)
        print("COSMOS DB ACCESS TESTS COMPLETED")
        print("=" * 80)
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

def test_specific_operations():
    """Test specific database operations"""
    try:
        print("\n" + "=" * 80)
        print("TESTING SPECIFIC COSMOS OPERATIONS")
        print("=" * 80)
        
        test_suite = CosmosTestSuite()
        
        # Add a test student
        print("\n1. ADDING TEST STUDENT")
        print("-" * 40)
        test_student = {
            "pen": "TEST123456",
            "legalFirstName": "TestFirst",
            "legalMiddleNames": "TestMiddle",
            "legalLastName": "TestLast",
            "dob": "2005-06-15",
            "localID": "TEST001"
        }
        
        # Generate embedding
        embedding = test_suite.embedding_service.generate_embedding(test_student)
        
        # Insert into Cosmos
        result = test_suite.cosmos_client.insert_student_embedding(test_student, embedding)
        print(f"Inserted test student: {result['id']}")
        
        # Test retrieval
        print("\n2. RETRIEVING TEST STUDENT")
        print("-" * 40)
        retrieved = test_suite.get_student_by_pen("TEST123456")
        if retrieved:
            test_suite.print_student_summary([retrieved])
        
        # Test name search
        print("\n3. SEARCHING BY TEST NAME")
        print("-" * 40)
        name_results = test_suite.get_students_by_exact_name("TestFirst", "TestLast")
        test_suite.print_student_summary(name_results)
        
        print("\nSpecific operations test completed")
        
    except Exception as e:
        print(f"Specific operations test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run general access tests
    test_cosmos_access()
    
    # Run specific operations tests
    test_specific_operations()