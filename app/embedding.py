from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import hashlib
import numpy as np
from test_API import StudentAPI  

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

        # Variant 1: legal names + dob
        parts1 = []
        for f in ["legalFirstName", "legalLastName", "dob"]:
            if student.get(f):
                parts1.append(student[f] if f != "dob" else f"born {student[f]}")
        variants["legal_name_dob"] = ", ".join(parts1)

        # Variant 2: usual names + dob
        parts2 = []
        for f in ["usualFirstName", "usualLastName", "dob"]:
            if student.get(f):
                parts2.append(student[f] if f != "dob" else f"born {student[f]}")
        variants["usual_name_dob"] = ", ".join(parts2)

        # Variant 3: legal names + grade
        parts3 = []
        for f in ["legalFirstName", "legalLastName", "gradeCode"]:
            if student.get(f):
                parts3.append(student[f] if f != "gradeCode" else f"grade {student[f]}")
        variants["legal_name_grade"] = ", ".join(parts3)

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
        """
        for student in students:
            pen = student.get("pen") or student.get("studentID")
            embeddings = self.embed_student_variant(student, model_name)
            self.index[pen] = {
                "embeddings": embeddings,
                "raw_data": {k: student.get(k) for k in [
                    "legalFirstName", "legalMiddleNames", "legalLastName",
                    "dob", "sexCode", "genderCode", "email",
                    "postalCode", "localID", "gradeCode", "gradeYear"
                ]}
            }

    # ------------------ Local search ------------------
    def search_local(self, query_student, model_name="MiniLM", variant="legal_name_dob", top_k=1):
        """
        Find top_k closest students in local index to query_student.
        Returns a list of raw_data dicts.
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

    # Test search for the first student
    query = students[0]  # pretend we want to find this student
    matches = emb.search_local(query, top_k=3)
    print("\nTop matches for query student:")
    for score, student_info in matches:
        print(f"Score: {score:.4f}, Student info: {student_info}")
