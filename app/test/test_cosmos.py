import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.cosmos_client import CosmosDBClient
from core.student_embedding import StudentEmbedding

class CosmosTestSuite:
    def __init__(self):
        self.cosmos_client = CosmosDBClient()
        self.embedding_service = StudentEmbedding()

    def get_all_students_paginated(self, page_size=20, max_pages=None):
        """Get all students with pagination"""
        query = "SELECT * FROM c ORDER BY c.pen"
        query_iterator = self.cosmos_client.container.query_items(
            query=query,
            enable_cross_partition_query=True,
            max_item_count=page_size
        )
        
        all_students = []
        page_count = 0
        
        for page in query_iterator.by_page():
            page_count += 1
            page_items = list(page)
            all_students.extend(page_items)
            
            if max_pages and page_count >= max_pages:
                break
        
        return all_students

    def get_students_by_exact_name(self, first_name, last_name):
        """Get students by exact first and last name"""
        return self.cosmos_client.get_students_by_name(first_name, last_name)

    def get_student_by_pen(self, pen):
        """Get specific student by PEN"""
        return self.cosmos_client.get_student_by_pen(pen)

    def get_database_statistics(self):
        """Get database statistics"""
        count_query = "SELECT VALUE COUNT(1) FROM c"
        total_count = list(self.cosmos_client.container.query_items(
            query=count_query,
            enable_cross_partition_query=True
        ))[0]
        
        return {"total_students": total_count}

    def print_student_summary(self, students, max_display=3):
        """Print summary of students"""
        if not students:
            print("No students found")
            return
        
        for i, student in enumerate(students[:max_display]):
            print(f"{i+1}. PEN: {student.get('pen')}, Name: {student.get('legalFirstName')} {student.get('legalLastName')}")

def test_cosmos_operations():
    """Test essential Cosmos operations"""
    try:
        print("TESTING COSMOS DB OPERATIONS")
        
        test_suite = CosmosTestSuite()
        
        # Database statistics
        stats = test_suite.get_database_statistics()
        print(f"Total students: {stats['total_students']}")
        
        # Get students by pages
        paginated_students = test_suite.get_all_students_paginated(page_size=5, max_pages=1)
        print(f"Retrieved {len(paginated_students)} students")
        test_suite.print_student_summary(paginated_students)
        
        # Test specific operations if students exist
        if paginated_students:
            # Get by PEN
            test_pen = paginated_students[0].get('pen')
            pen_student = test_suite.get_student_by_pen(test_pen)
            print(f"Found student by PEN: {pen_student.get('legalFirstName') if pen_student else 'None'}")
            
            # Get by exact name
            test_student = paginated_students[0]
            first_name = test_student.get('legalFirstName')
            last_name = test_student.get('legalLastName')
            exact_name_students = test_suite.get_students_by_exact_name(first_name, last_name)
            print(f"Found {len(exact_name_students)} students with name {first_name} {last_name}")
        
        print("Cosmos operations test completed")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_cosmos_operations()