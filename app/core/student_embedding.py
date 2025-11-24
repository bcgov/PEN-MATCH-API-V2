import numpy as np
from openai import OpenAI, AzureOpenAI
from config.settings import settings

class StudentEmbedding:
    def __init__(self):
        # Configure OpenAI client
        if settings.openai_api_base_embedding:
            self.openai_client = AzureOpenAI(
                api_key=settings.openai_api_key,
                api_version="2023-05-15",
                azure_endpoint=settings.openai_api_base_embedding
            )
        else:
            self.openai_client = OpenAI(api_key=settings.openai_api_key)

    def student_to_text(self, student):
        """Convert ONLY student name data to text for embedding"""
        parts = []
        
        # First name
        if student.get("legalFirstName") and student["legalFirstName"] != 'NULL':
            parts.append(f"First name: {student['legalFirstName']}.")
        
        # Last name
        if student.get("legalLastName") and student["legalLastName"] != 'NULL':
            parts.append(f"Last name: {student['legalLastName']}.")
        
        # Middle name
        if student.get("legalMiddleNames") and student["legalMiddleNames"] != 'NULL':
            parts.append(f"Middle name: {student['legalMiddleNames']}.")
        
        # ONLY embed names - other fields will be stored as separate columns
        return " ".join(parts)

    def generate_embedding(self, student):
        """Generate embedding for student names only"""
        text = self.student_to_text(student)
        
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            raise ValueError(f"Failed to generate embedding: {str(e)}")

    def prepare_student_data(self, student):
        """Prepare student data with embedding and separate columns"""
        embedding = self.generate_embedding(student)
        
        return {
            "pen": student.get("pen"),
            "embedding": embedding,
            "dob": student.get("dob"),
            "postal_code": student.get("postalCode"),
            "mincode": student.get("mincode"),
            "sex_code": student.get("sexCode"),
            "student_data": {
                "pen": student.get("pen"),
                "legalFirstName": student.get("legalFirstName"),
                "legalLastName": student.get("legalLastName"),
                "legalMiddleNames": student.get("legalMiddleNames"),
                "dob": student.get("dob"),
                "sexCode": student.get("sexCode"),
                "postalCode": student.get("postalCode"),
                "mincode": student.get("mincode"),
                "localID": student.get("localID")
            }
        }

    def generate_embeddings_batch(self, students):
        """Generate embeddings for multiple students with separate columns"""
        embeddings = {}
        for student in students:
            pen = student.get("pen")
            if pen:
                embeddings[pen] = self.prepare_student_data(student)
        return embeddings
    
if __name__ == "__main__":
    try:
        embedding_service = StudentEmbedding()
        print("Testing StudentEmbedding...")
        
        # Test with sample student data
        sample_students = [
            {
                "pen": "123456789",
                "legalFirstName": "John",
                "legalLastName": "Doe",
                "legalMiddleNames": "Michael",
                "dob": "1995-09-20",
                "sexCode": "M",
                "postalCode": "V5K2A1",
                "mincode": "12345678",
                "localID": "STU001"
            },
            {
                "pen": "987654321",
                "legalFirstName": "Jane",
                "legalLastName": "Smith",
                "dob": "2005-03-15",
                "sexCode": "F",
                "postalCode": "V6B1A1",
                "mincode": "87654321",
                "localID": "STU002"
            }
        ]
        
        # Test text conversion for each student (names only)
        for i, student in enumerate(sample_students, 1):
            text = embedding_service.student_to_text(student)
            print(f"Student {i} text for embedding: {text}")
        
        # Test single embedding generation
        embedding = embedding_service.generate_embedding(sample_students[0])
        print(f"Generated embedding with {len(embedding)} dimensions")
        
        # Test prepared student data
        prepared_data = embedding_service.prepare_student_data(sample_students[0])
        print(f"Prepared data structure:")
        print(f"  PEN: {prepared_data['pen']}")
        print(f"  DOB: {prepared_data['dob']}")
        print(f"  Postal Code: {prepared_data['postal_code']}")
        print(f"  Mincode: {prepared_data['mincode']}")
        print(f"  Sex Code: {prepared_data['sex_code']}")
        print(f"  Embedding dimensions: {len(prepared_data['embedding'])}")
        
        # Test batch embedding generation
        embeddings = embedding_service.generate_embeddings_batch(sample_students)
        print(f"Generated embeddings for {len(embeddings)} students")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")