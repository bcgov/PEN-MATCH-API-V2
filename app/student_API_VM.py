import os
import requests
from azure.keyvault.secrets import SecretClient
from azure.identity import ManagedIdentityCredential
from openai import OpenAI, AzureOpenAI
import json

class StudentAPI:
    def __init__(self, key_vault_url=None):
        # Initialize Azure Key Vault client if URL provided
        if key_vault_url:
            credential = ManagedIdentityCredential()
            self.secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
            
            # Retrieve secrets from Azure Key Vault
            self.tenant_url = self.get_secret("TENANT-URL")
            self.client_id = self.get_secret("CLIENT-ID")
            self.client_secret = self.get_secret("CLIENT-SECRET")
            self.api_base_url = self.get_secret("API-BASE-URL")
            
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
            self.tenant_url = os.getenv("TENANT_URL")
            self.client_id = os.getenv("CLIENT_ID")
            self.client_secret = os.getenv("CLIENT_SECRET")
            self.api_base_url = os.getenv("API_BASE_URL")
            
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

    # ------------------ Authentication ------------------
    def get_access_token(self):
        token_url = f"{self.tenant_url}/protocol/openid-connect/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise ValueError("No access_token returned")
        return token

    # ------------------ Fetch Students ------------------
    def get_student_page(self, page=1, size=20, sort=None, filter_query=None):
        token = self.get_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        params = {"page": page, "size": size}
        if sort:
            params["sort"] = sort
        if filter_query:
            params["filter"] = filter_query

        endpoint = f"{self.api_base_url}/api/v1/student/paginated"
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        try:
            data = response.json()
        except json.JSONDecodeError:
            raise ValueError("API did not return valid JSON")

        if isinstance(data, dict) and "content" in data:
            return data["content"]
        elif isinstance(data, list):
            return data
        else:
            return [data]

    @staticmethod
    def print_json_structure(data, indent=0):
        prefix = "  " * indent
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{prefix}{key}: {type(value).__name__}")
                if isinstance(value, (dict, list)):
                    StudentAPI.print_json_structure(value, indent + 1)
        elif isinstance(data, list):
            print(f"{prefix}List[{len(data)}]:")
            if len(data) > 0:
                StudentAPI.print_json_structure(data[0], indent + 1)
        else:
            print(f"{prefix}{data} ({type(data).__name__})")

    @staticmethod
    def print_student_info(students):
        for idx, student in enumerate(students, start=1):
            if not isinstance(student, dict):
                print(idx, student)
                continue
            
            # Collect student information
            pen = student.get("pen", "N/A")
            first_name = student.get("legalFirstName", "")
            middle_names = student.get("legalMiddleNames", "")
            last_name = student.get("legalLastName", "")
            dob = student.get("dob", "N/A")
            local_id = student.get("localID", "N/A")
            
            # Build full name
            name_parts = [first_name]
            if middle_names:
                name_parts.append(middle_names)
            name_parts.append(last_name)
            full_name = " ".join(filter(None, name_parts))
            
            print(f"{idx}. {full_name}")
            print(f"    PEN: {pen}")
            print(f"    DOB: {dob}")
            print(f"    Local ID: {local_id}")

    @staticmethod
    def print_one_student(student, indent=2):
        if not isinstance(student, dict):
            print("Student is not a dictionary:", student)
            return
        print("Student Record:")
        for key, value in student.items():
            print(" " * indent + f"{key}: {value}")

    # ------------------ Embedding Utilities ------------------
    @staticmethod
    def student_to_text(student):
        parts = []
        
        # Include PEN
        if student.get("pen"):
            parts.append(f"PEN {student['pen']}")
            
        # Include full name
        if student.get("legalFirstName"):
            parts.append(student["legalFirstName"])
        if student.get("legalMiddleNames"):
            parts.append(student["legalMiddleNames"])
        if student.get("legalLastName"):
            parts.append(student["legalLastName"])
            
        # Include DOB
        if student.get("dob"):
            parts.append(f"born {student['dob']}")
            
        # Include Local ID
        if student.get("localID"):
            parts.append(f"local ID {student['localID']}")
            
        return ", ".join(parts)

    def embed_student(self, student):
        """Generate embedding using OpenAI's text-embedding-ada-002 model"""
        text = self.student_to_text(student)
        
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            
            embedding = response.data[0].embedding
            return embedding
            
        except Exception as e:
            raise ValueError(f"Failed to generate embedding: {str(e)}")

# ------------------ Example usage ------------------
if __name__ == "__main__":
    # Option 1: Use Azure Key Vault with Managed Identity
    key_vault_url = "https://pen-match-api-v2.vault.azure.net"
    api = StudentAPI(key_vault_url=key_vault_url)
    
    # Option 2: Use environment variables (fallback)
    # api = StudentAPI()

    # Fetch and inspect students
    students = api.get_student_page(page=1, size=10)
    print("JSON structure of API response:")
    api.print_json_structure(students)
    api.print_student_info(students)
    api.print_one_student(students[0])

    # Test embedding one student
    sample_student = {
        "pen": 350800,
        "legalFirstName": "ROBYN",
        "legalMiddleNames": "DONNELLY",
        "legalLastName": "ANDERSON",
        "dob": "2010-11-09",
        "localID": "892056259223"
    }
    vector = api.embed_student(sample_student)
    print("Embedding vector length:", len(vector))
    print("First 10 dimensions:", vector[:10])