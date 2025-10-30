from azure.keyvault.secrets import SecretClient
from azure.identity import ManagedIdentityCredential
from openai import OpenAI, AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import json
import hashlib
import numpy as np
import os
import requests
from student_API_VM import StudentAPI  

class StudentEmbedding:
    def __init__(self, student_api, key_vault_url=None):
        self.api = student_api
        self.index = {}  # will store pen -> embeddings + raw data
        
        # Initialize Azure Key Vault client if URL provided
        if key_vault_url:
            credential = ManagedIdentityCredential()
            self.secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
            
            # OpenAI configuration from Key Vault
            self.openai_api_key = self.get_secret("OPENAI-API-KEY")
            try:
                self.openai_api_base = self.get_secret("OPENAI-API-BASE")  # Optional: for Azure OpenAI
            except:
                self.openai_api_base = None
            
            # Configure OpenAI client
            if self.openai_api_base:
                # Azure OpenAI
                self.openai_client = AzureOpenAI(
                    api_key=self.openai_api_key,
                    api_version="2023-05-15",
                    azure_endpoint=self.openai_api_base
                )
            else:
                # Standard OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            # Fallback to environment variables
            from dotenv import load_dotenv
            load_dotenv()
            
            # OpenAI configuration from environment
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.openai_api_base = os.getenv("OPENAI_API_BASE")
            
            if self.openai_api_base:
                # Azure OpenAI
                self.openai_client = AzureOpenAI(
                    api_key=self.openai_api_key,
                    api_version="2023-05-15",
                    azure_endpoint=self.openai_api_base
                )
            else:
                # Standard OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)

    def get_secret(self, secret_name):
        """Retrieve a secret from Azure Key Vault"""
        try:
            retrieved_secret = self.secret_client.get_secret(secret_name)
            return retrieved_secret.value
        except Exception as e:
            raise ValueError(f"Failed to retrieve secret '{secret_name}' from Key Vault: {str(e)}")

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
    def embed_student_variant(self, student):
        """Generate embedding using OpenAI's text-embedding-ada-002 model"""
        variants = self.student_to_text_variants(student)
        embeddings = {}
        
        for variant_name, text in variants.items():
            try:
                response = self.openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                
                embedding = response.data[0].embedding
                embeddings[variant_name] = np.array(embedding)
                
            except Exception as e:
                raise ValueError(f"Failed to generate embedding for variant '{variant_name}': {str(e)}")
        
        return embeddings

    # ------------------ Build local index ------------------
    def build_local_index(self, students):
        """
        Stores embeddings for each student in a local dict using pen as key.
        Only stores the specified core fields.
        """
        for student in students:
            pen = student.get("pen")
            if not pen:
                continue  # Skip students without PEN
                
            embeddings = self.embed_student_variant(student)
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
    def search_local(self, query_student, variant="student_info", top_k=1):
        """
        Find top_k closest students in local index to query_student.
        Returns a list of raw_data dicts with only core fields.
        """
        # Generate query embedding using OpenAI
        query_embeddings = self.embed_student_variant(query_student)
        query_vec = query_embeddings[variant].reshape(1, -1)

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
    # Use Azure Key Vault with Managed Identity
    key_vault_url = "https://pen-match-api-v2.vault.azure.net"
    
    # Initialize API and embedding system with same credentials
    api = StudentAPI(key_vault_url=key_vault_url)
    emb = StudentEmbedding(api, key_vault_url=key_vault_url)
    
    students = api.get_student_page(page=1, size=10)  # first 10 students
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