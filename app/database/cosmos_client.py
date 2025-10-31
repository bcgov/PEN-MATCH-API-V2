from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceNotFoundError, CosmosResourceExistsError
from app.config.settings import settings
import json

class CosmosDBClient:
    def __init__(self):
        self.client = CosmosClient(settings.cosmos_endpoint, settings.cosmos_key)
        self.database_name = "student_embeddings"
        self.container_name = "student_records"
        self._setup_database()

    def _setup_database(self):
        """Setup database and container with proper indexing"""
        # Create database
        self.database = self.client.create_database_if_not_exists(id=self.database_name)
        
        # Create container with partition key and indexing policy
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [
                {"path": "/*"}
            ],
            "excludedPaths": [
                {"path": "/embedding/*"}  # Exclude embedding from indexing
            ],
            "compositeIndexes": [
                [
                    {"path": "/legalFirstName", "order": "ascending"},
                    {"path": "/legalLastName", "order": "ascending"}
                ]
            ]
        }
        
        self.container = self.database.create_container_if_not_exists(
            id=self.container_name,
            partition_key=PartitionKey(path="/pen"),
            indexing_policy=indexing_policy
        )

    def insert_student_embedding(self, student_data, embedding):
        """Insert student with embedding into Cosmos DB"""
        document = {
            "id": str(student_data["pen"]),
            "pen": student_data["pen"],
            "legalFirstName": student_data.get("legalFirstName", ""),
            "legalMiddleNames": student_data.get("legalMiddleNames", ""),
            "legalLastName": student_data.get("legalLastName", ""),
            "dob": student_data.get("dob", ""),
            "localID": student_data.get("localID", ""),
            "embedding": embedding,
            "name_key": f"{student_data.get('legalFirstName', '')}_{student_data.get('legalLastName', '')}"
        }
        
        try:
            return self.container.create_item(body=document)
        except CosmosResourceExistsError:
            # If item already exists, replace it
            return self.container.replace_item(item=document["id"], body=document)

    def get_students_by_name(self, first_name, last_name):
        """Get students by first and last name"""
        query = """
        SELECT * FROM c 
        WHERE c.legalFirstName = @first_name 
        AND c.legalLastName = @last_name
        """
        parameters = [
            {"name": "@first_name", "value": first_name},
            {"name": "@last_name", "value": last_name}
        ]
        
        items = list(self.container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        return items

    def get_student_by_pen(self, pen):
        """Get student by PEN"""
        try:
            return self.container.read_item(item=str(pen), partition_key=pen)
        except CosmosResourceNotFoundError:
            return None

    def name_exists(self, first_name, last_name):
        """Check if name combination exists in database"""
        students = self.get_students_by_name(first_name, last_name)
        return len(students) > 0

    def batch_insert_embeddings(self, embeddings_dict):
        """Insert multiple student embeddings"""
        results = []
        for pen, data in embeddings_dict.items():
            try:
                result = self.insert_student_embedding(
                    data["student_data"], 
                    data["embedding"]
                )
                results.append(result)
                print(f"Successfully inserted student {pen}")
            except Exception as e:
                print(f"Failed to insert student {pen}: {str(e)}")
        return results
    
if __name__ == "__main__":
    try:
        cosmos_client = CosmosDBClient()
        print("Testing CosmosDBClient...")
        
        # Test data
        sample_student = {
            "pen": "123456789",
            "legalFirstName": "John",
            "legalMiddleNames": "Michael",
            "legalLastName": "Doe",
            "dob": "2005-01-15",
            "localID": "STU001"
        }
        sample_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 300  # Mock embedding
        
        # Test insert
        print("Testing insert...")
        result = cosmos_client.insert_student_embedding(sample_student, sample_embedding)
        print(f"Inserted student: {result['id']}")
        
        # Test get by PEN
        print("Testing get by PEN...")
        student = cosmos_client.get_student_by_pen("123456789")
        if student:
            print(f"Found student: {student['legalFirstName']} {student['legalLastName']}")
        
        # Test get by name
        print("Testing get by name...")
        students = cosmos_client.get_students_by_name("John", "Doe")
        print(f"Found {len(students)} students with name John Doe")
        
        # Test name exists
        exists = cosmos_client.name_exists("John", "Doe")
        print(f"Name exists: {exists}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")