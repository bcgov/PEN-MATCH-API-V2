import numpy as np
from openai import OpenAI, AzureOpenAI
from config.settings import settings

class StudentEmbedding:
    def __init__(self):
        # Configure OpenAI client
        if settings.openai_api_base_embedding:
            self.openai_client = AzureOpenAI(
                api_key=settings.openai_api_key,
                api_version="2025-01-01",
                azure_endpoint=settings.openai_api_base_embedding_3
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
        if student.get("legalLastName"):
            parts.append(student["legalLastName"])
        if student.get("legalMiddleNames"):
            parts.append(student["legalMiddleNames"])
        if student.get("dob"):
            parts.append(f"born {student['dob']}")
        if student.get("sexCode"):
            parts.append(f"sex {student['sexCode']}")
        if student.get("postalCode"):
            parts.append(f"postal code {student['postalCode']}")
        if student.get("mincode"):
            parts.append(f"mincode {student['mincode']}")
        if student.get("localID"):
            parts.append(f"local ID {student['localID']}")
        
        return ", ".join(parts)

    def generate_embedding(self, student):
        """Generate embedding for a student"""
        text = self.student_to_text(student)
        
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            raise ValueError(f"Failed to generate embedding: {str(e)}")

    def generate_embeddings_batch(self, students):
        """Generate embeddings for multiple students"""
        embeddings = {}
        for student in students:
            # Use pen as primary index
            pen = student.get("pen")
            if pen:
                embeddings[pen] = {
                    "embedding": self.generate_embedding(student),
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
        return embeddings
    
if __name__ == "__main__":
    try:
        embedding_service = StudentEmbedding()
        print("Testing StudentEmbedding...")
        
        # Test with sample student data
        sample_student = {
            "pen": "123456789",
            "legalFirstName": "John",
            "legalLastName": "Doe",
            "legalMiddleNames": "Michael",
            "dob": "2005-01-15",
            "sexCode": "M",
            "postalCode": "V5K2A1",
            "mincode": "12345678",
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
            "dob": "2004-03-20",
            "sexCode": "F",
            "postalCode": "V6B1A1",
            "mincode": "87654321",
            "localID": "STU002"
        }]
        
        embeddings = embedding_service.generate_embeddings_batch(students)
        print(f"Generated embeddings for {len(embeddings)} students")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")