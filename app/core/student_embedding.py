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
        """Convert student data to text for embedding"""
        parts = []
        
        if student.get("studentID"):
            parts.append(f"Student ID {student['studentID']}")
        if student.get("pen"):
            parts.append(f"PEN {student['pen']}")
        if student.get("legalFirstName"):
            parts.append(student["legalFirstName"])
        if student.get("legalMiddleNames"):
            parts.append(student["legalMiddleNames"])
        if student.get("legalLastName"):
            parts.append(student["legalLastName"])
        if student.get("usualFirstName"):
            parts.append(f"usual first name {student['usualFirstName']}")
        if student.get("usualMiddleNames"):
            parts.append(f"usual middle names {student['usualMiddleNames']}")
        if student.get("usualLastName"):
            parts.append(f"usual last name {student['usualLastName']}")
        if student.get("dob"):
            parts.append(f"born {student['dob']}")
        if student.get("sexCode"):
            parts.append(f"sex {student['sexCode']}")
        if student.get("deceasedDate"):
            parts.append(f"deceased {student['deceasedDate']}")
        if student.get("postalCode"):
            parts.append(f"postal code {student['postalCode']}")
        if student.get("mincode"):
            parts.append(f"mincode {student['mincode']}")
        if student.get("localID"):
            parts.append(f"local ID {student['localID']}")
        if student.get("gradeCode"):
            parts.append(f"grade {student['gradeCode']}")
        if student.get("demogCode"):
            parts.append(f"demog {student['demogCode']}")
        if student.get("statusCode"):
            parts.append(f"status {student['statusCode']}")
        
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
            # Use studentID as primary index, fallback to pen
            student_id = student.get("studentID") or student.get("pen")
            if student_id:
                embeddings[student_id] = {
                    "embedding": self.generate_embedding(student),
                    "student_data": {
                        "studentID": student.get("studentID"),
                        "pen": student.get("pen"),
                        "legalFirstName": student.get("legalFirstName"),
                        "legalMiddleNames": student.get("legalMiddleNames"),
                        "legalLastName": student.get("legalLastName"),
                        "usualFirstName": student.get("usualFirstName"),
                        "usualMiddleNames": student.get("usualMiddleNames"),
                        "usualLastName": student.get("usualLastName"),
                        "dob": student.get("dob"),
                        "sexCode": student.get("sexCode"),
                        "deceasedDate": student.get("deceasedDate"),
                        "postalCode": student.get("postalCode"),
                        "mincode": student.get("mincode"),
                        "localID": student.get("localID"),
                        "gradeCode": student.get("gradeCode"),
                        "demogCode": student.get("demogCode"),
                        "statusCode": student.get("statusCode")
                    }
                }
        return embeddings
    
if __name__ == "__main__":
    try:
        embedding_service = StudentEmbedding()
        print("Testing StudentEmbedding...")
        
        # Test with sample student data
        sample_student = {
            "studentID": "STU123456789",
            "pen": "123456789",
            "legalFirstName": "John",
            "legalMiddleNames": "Michael",
            "legalLastName": "Doe",
            "usualFirstName": "Johnny",
            "usualMiddleNames": "Mike",
            "usualLastName": "Doe",
            "dob": "2005-01-15",
            "sexCode": "M",
            "postalCode": "V5K2A1",
            "mincode": "12345678",
            "localID": "STU001",
            "gradeCode": "10",
            "demogCode": "A",
            "statusCode": "A"
        }
        
        # Test text conversion
        text = embedding_service.student_to_text(sample_student)
        print(f"Student text: {text}")
        
        # Test single embedding generation
        embedding = embedding_service.generate_embedding(sample_student)
        print(f"Generated embedding with {len(embedding)} dimensions")
        
        # Test batch embedding generation
        students = [sample_student, {
            "studentID": "STU987654321",
            "pen": "987654321",
            "legalFirstName": "Jane",
            "legalLastName": "Smith",
            "dob": "2004-03-20",
            "sexCode": "F",
            "gradeCode": "11"
        }]
        
        embeddings = embedding_service.generate_embeddings_batch(students)
        print(f"Generated embeddings for {len(embeddings)} students")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")