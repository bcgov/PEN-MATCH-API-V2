from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import hashlib
import numpy as np
from student_API import StudentAPI  

class StudentEmbedding:
    def __init__(self, student_api):
        self.api = student_api
        self.embedding_models = {
            "MiniLM": SentenceTransformer("all-MiniLM-L6-v2"),
        }
        self.index = {}  # will store pen -> embeddings + raw data

    # ------------------ Generate text variants ------------------
    def student_to_text_variants(self, student):
        variants = {}

        # Single variant: only pen + legal names + dob + localID
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
        
        variants["student_info"] = ", ".join(parts)
        return variants

    # ------------------ Generate embedding ------------------
    def embed_student_variant(self, student, model_name="MiniLM"):
        if model_name not in self.embedding_models:
            raise ValueError(f"Model {model_name} not available")
        model = self.embedding_models[model_name]
        variants = self.student_to_text_variants(student)
        embeddings = {k: model.encode(v) for k, v in variants.items()}
        return embeddings

    # ------------------ Build local index ------------------
    def build_local_index(self, students, model_name="MiniLM"):
        """
        Stores embeddings for each student in a local dict using pen as key.
        Only stores the specified core fields.
        """
        for student in students:
            pen = student.get("pen")
            if not pen:
                continue  # Skip students without PEN
                
            embeddings = self.embed_student_variant(student, model_name)
            self.index[pen] = {
                "embeddings": embeddings,
                "raw_data": {
                    "pen": student.get("pen"),
                    "legalFirstName": student.get("legalFirstName"),
                    "legalMiddleNames": student.get("legalMiddleNames"),
                    "legalLastName": student.get("legalLastName"),
                    "dob": student.get("dob"),
                    "localID": student.get("localID")
                }
            }

    # ------------------ Local search ------------------
    def search_local(self, query_student, model_name="MiniLM", variant="student_info", top_k=1):
        """
        Find top_k closest students in local index to query_student.
        Returns a list of raw_data dicts with only core fields.
        """
        if model_name not in self.embedding_models:
            raise ValueError(f"Model {model_name} not available")
        model = self.embedding_models[model_name]

        # Generate query embedding
        query_text = self.student_to_text_variants(query_student).get(variant, "")
        query_vec = model.encode(query_text).reshape(1, -1)

        # Compare with all students
        results = []
        for pen, info in self.index.items():
            candidate_vec = info["embeddings"][variant].reshape(1, -1)
            score = cosine_similarity(query_vec, candidate_vec)[0][0]
            results.append((score, info["raw_data"]))

        # Sort by similarity
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]


# ------------------ Example usage ------------------
if __name__ == "__main__":
    api = StudentAPI()
    students = api.get_student_page(page=1, size=10)  # first 10 students

    emb = StudentEmbedding(api)
    emb.build_local_index(students)

    # Test search with the provided student structure
    query = {
        "pen": 350800355,
        "legalFirstName": "ROBYN",
        "legalMiddleNames": "DONNELLY",
        "legalLastName": "ANDERSON",
        "dob": "2010-11-09",
        "localID": "892056259223"
    }
    
    matches = emb.search_local(query, top_k=3)
    print("\nTop matches for query student:")
    for score, student_info in matches:
        print(f"Score: {score:.4f}")
        print(f"  PEN: {student_info.get('pen', 'N/A')}")
        print(f"  First Name: {student_info.get('legalFirstName', 'N/A')}")
        print(f"  Middle Names: {student_info.get('legalMiddleNames', 'N/A')}")
        print(f"  Last Name: {student_info.get('legalLastName', 'N/A')}")
        print(f"  DOB: {student_info.get('dob', 'N/A')}")
        print(f"  Local ID: {student_info.get('localID', 'N/A')}")
        print()