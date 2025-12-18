from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from database.student_api import StudentAPI
from core.student_embedding import StudentEmbedding
from database.cosmos_client import CosmosDBClient

class StudentWorkflow:
    def __init__(self):
        self.student_api = StudentAPI()
        self.embedding_service = StudentEmbedding()
        self.cosmos_client = CosmosDBClient()
        self.similarity_threshold = 0.95  # Threshold for perfect match

    def create_embeddings_for_students(self, students):
        """Create embeddings and store in Cosmos DB"""
        print(f"Creating embeddings for {len(students)} students")
        
        # Generate embeddings
        embeddings = self.embedding_service.generate_embeddings_batch(students)
        
        # Store in Cosmos DB
        results = self.cosmos_client.batch_insert_embeddings(embeddings)
        
        print(f"Successfully stored {len(results)} student embeddings")
        return results

    def find_perfect_match(self, query_student, candidates):
        """Find perfect match using embedding similarity"""
        query_embedding = self.embedding_service.generate_embedding(query_student)
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        best_match = None 
        best_score = 0
        
        for candidate in candidates:
            candidate_vec = np.array(candidate["embedding"]).reshape(1, -1)
            score = cosine_similarity(query_vec, candidate_vec)[0][0]
            
            if score > best_score:
                best_score = score
                best_match = candidate
        
        # Check if it's a perfect match
        if best_score >= self.similarity_threshold:
            print(f"Perfect match found with score: {best_score}")
            return best_match, best_score
        
        print(f"No perfect match found. Best score: {best_score}")
        return None, best_score

    def process_student_query(self, query_student):
        """Main workflow for processing student query"""
        first_name = query_student.get("legalFirstName")
        last_name = query_student.get("legalLastName")
        
        if not first_name or not last_name:
            raise ValueError("First name and last name are required")
        
        print(f"Processing query for: {first_name} {last_name}")
        
        # Step 1: Check if name exists in Cosmos DB
        if self.cosmos_client.name_exists(first_name, last_name):
            print("Name found in Cosmos DB, fetching candidates")
            
            # Fetch candidates from Cosmos
            candidates = self.cosmos_client.get_students_by_name(first_name, last_name)
            
            # Find perfect match
            perfect_match, score = self.find_perfect_match(query_student, candidates)
            
            if perfect_match:
                return {
                    "status": "perfect_match_found",
                    "student": perfect_match,
                    "similarity_score": score,
                    "source": "cosmos_db"
                }
            else:
                return {
                    "status": "no_perfect_match",
                    "candidates": candidates,
                    "best_score": score,
                    "source": "cosmos_db",
                    "next_action": "further_analysis_needed"
                }
        
        else:
            print("Name not found in Cosmos DB, fetching from source")
            
            # Step 2: Fetch from source database
            source_students = self.student_api.get_students_by_name(first_name, last_name)
            
            if not source_students:
                return {
                    "status": "no_students_found",
                    "message": f"No students found with name: {first_name} {last_name}"
                }
            
            # Step 3: Create embeddings and store in Cosmos
            self.create_embeddings_for_students(source_students)
            
            # Step 4: Fetch from Cosmos and find match
            candidates = self.cosmos_client.get_students_by_name(first_name, last_name)
            perfect_match, score = self.find_perfect_match(query_student, candidates)
            
            if perfect_match:
                return {
                    "status": "perfect_match_found",
                    "student": perfect_match,
                    "similarity_score": score,
                    "source": "source_database_then_cosmos"
                }
            else:
                return {
                    "status": "no_perfect_match",
                    "candidates": candidates,
                    "best_score": score,
                    "source": "source_database_then_cosmos",
                    "next_action": "further_analysis_needed"
                }

    def bulk_import_students(self, page_size=100, max_pages=None):
        """Bulk import students from source to Cosmos DB"""
        page = 1
        total_imported = 0
        
        while True:
            if max_pages and page > max_pages:
                break
                
            print(f"Processing page {page}")
            students = self.student_api.get_student_page(page=page, size=page_size)
            
            if not students:
                print("No more students to process")
                break
            
            # Filter students that don't exist in Cosmos
            new_students = []
            for student in students:
                pen = student.get("pen")
                if pen and not self.cosmos_client.get_student_by_pen(pen):
                    new_students.append(student)
            
            if new_students:
                self.create_embeddings_for_students(new_students)
                total_imported += len(new_students)
                print(f"Imported {len(new_students)} new students from page {page}")
            
            page += 1
        
        print(f"Bulk import completed. Total imported: {total_imported}")
        return total_imported

if __name__ == "__main__":
    try:
        workflow = StudentWorkflow()
        print("Testing StudentWorkflow...")
        
        # Test with sample query student
        query_student = {
            "pen": "987654321",
            "legalFirstName": "Jane",
            "legalMiddleNames": "",
            "legalLastName": "Smith",
            "dob": "2004-03-20",
            "localID": "STU002"
        }
        
        print(f"Processing query for: {query_student['legalFirstName']} {query_student['legalLastName']}")
        
        # Test student query processing
        result = workflow.process_student_query(query_student)
        print(f"Query result: {result['status']}")
        
        if result['status'] == 'perfect_match_found':
            print(f"Match found with score: {result['similarity_score']:.4f}")
        elif result['status'] == 'no_perfect_match':
            print(f"No perfect match. Best score: {result['best_score']:.4f}")
            print(f"Found {len(result['candidates'])} candidates")
        elif result['status'] == 'no_students_found':
            print(f"No students found: {result['message']}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")