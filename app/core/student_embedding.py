import numpy as np
from openai import OpenAI, AzureOpenAI
from config.settings import settings

class StudentEmbedding:
    def __init__(self):
        # Configure OpenAI client
        if settings.openai_api_base:
            self.openai_client = AzureOpenAI(
                api_key=settings.openai_api_key,
                api_version="2023-05-15",
                azure_endpoint=settings.openai_api_base
            )
        else:
            self.openai_client = OpenAI(api_key=settings.openai_api_key)

    def student_to_text(self, student):
        """Convert student data to text for embedding"""
        parts = []
        
        if student.get("pen"):
            parts.append(f"PEN {student['pen']}")
        if student.get("legalFirstName"):
            parts.append(student["legalFirstName"])
        if student.get("legalMiddleNames"):
            parts.append(student["legalMiddleNames"])
        if student.get("legalLastName"):
            parts.append(student["legalLastName"])
        if student.get("dob"):
            parts.append(f"born {student['dob']}")
        if student.get("localID"):
            parts.append(f"local ID {student['localID']}")
        
        return ", ".join(parts)

    def generate_embedding(self, student):
        """Generate embedding for a student"""
        text = self.student_to_text(student)
        
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            raise ValueError(f"Failed to generate embedding: {str(e)}")

    def generate_embeddings_batch(self, students):
        """Generate embeddings for multiple students"""
        embeddings = {}
        for student in students:
            pen = student.get("pen")
            if pen:
                embeddings[pen] = {
                    "embedding": self.generate_embedding(student),
                    "student_data": {
                        "pen": student.get("pen"),
                        "legalFirstName": student.get("legalFirstName"),
                        "legalMiddleNames": student.get("legalMiddleNames"),
                        "legalLastName": student.get("legalLastName"),
                        "dob": student.get("dob"),
                        "localID": student.get("localID")
                    }
                }
        return embeddings
    
if __name__ == "__main__":
    try:
        embedding_service = StudentEmbedding()
        print("Testing StudentEmbedding...")
        
        # Test with sample student data
        sample_student = {
            "pen": "123456789",
            "legalFirstName": "John",
            "legalMiddleNames": "Michael",
            "legalLastName": "Doe",
            "dob": "2005-01-15",
            "localID": "STU001"
        }
        
        # Test text conversion
        text = embedding_service.student_to_text(sample_student)
        print(f"Student text: {text}")
        
        # Test single embedding generation
        embedding = embedding_service.generate_embedding(sample_student)
        print(f"Generated embedding with {len(embedding)} dimensions")
        
        # Test batch embedding generation
        students = [sample_student, {
            "pen": "987654321",
            "legalFirstName": "Jane",
            "legalLastName": "Smith",
            "dob": "2004-03-20"
        }]
        
        embeddings = embedding_service.generate_embeddings_batch(students)
        print(f"Generated embeddings for {len(embeddings)} students")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")