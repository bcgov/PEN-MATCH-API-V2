import requests
import json
from openai import OpenAI, AzureOpenAI
from config.settings import settings

class StudentAPI:
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

    def get_access_token(self):
        """Get access token for student API"""
        token_url = f"{settings.tenant_url}/protocol/openid-connect/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": settings.client_id,
            "client_secret": settings.client_secret
        }
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        token = response.json().get("access_token")
        if not token:
            raise ValueError("No access_token returned")
        return token

    def get_student_page(self, page=1, size=20, sort=None, filter_query=None):
        """Fetch paginated student data"""
        token = self.get_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        params = {"page": page, "size": size}
        if sort:
            params["sort"] = sort
        if filter_query:
            params["filter"] = filter_query

        endpoint = f"{settings.api_base_url}/api/v1/student/paginated"
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

    def get_students_by_name(self, first_name, last_name):
        """Get students by first and last name"""
        filter_query = f"legalFirstName eq '{first_name}' and legalLastName eq '{last_name}'"
        return self.get_student_page(filter_query=filter_query, size=100)

    @staticmethod
    def print_student_info(students):
        """Print formatted student information"""
        for idx, student in enumerate(students, start=1):
            if not isinstance(student, dict):
                print(idx, student)
                continue
            
            pen = student.get("pen", "N/A")
            first_name = student.get("legalFirstName", "")
            middle_names = student.get("legalMiddleNames", "")
            last_name = student.get("legalLastName", "")
            dob = student.get("dob", "N/A")
            local_id = student.get("localID", "N/A")
            
            name_parts = [first_name]
            if middle_names:
                name_parts.append(middle_names)
            name_parts.append(last_name)
            full_name = " ".join(filter(None, name_parts))
            
            print(f"{idx}. {full_name}")
            print(f"    PEN: {pen}")
            print(f"    DOB: {dob}")
            print(f"    Local ID: {local_id}")

if __name__ == "__main__":
    try:
        api = StudentAPI()
        print("Testing StudentAPI...")
        
        # Test getting a page of students
        students = api.get_student_page(page=1, size=5)
        print(f"Retrieved {len(students)} students")
        
        if students:
            print("\nFirst student:")
            api.print_student_info([students[0]])
            
            # Test search by name if we have student data
            first_student = students[0]
            if isinstance(first_student, dict):
                first_name = first_student.get("legalFirstName")
                last_name = first_student.get("legalLastName")
                if first_name and last_name:
                    print(f"\nSearching for students named {first_name} {last_name}...")
                    search_results = api.get_students_by_name(first_name, last_name)
                    print(f"Found {len(search_results)} matching students")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")