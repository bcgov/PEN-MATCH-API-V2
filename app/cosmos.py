"""
Simple Azure Cosmos DB Connection Test
Minimal test script that mimics the OpenAI test pattern
"""

import os
import json
import uuid
from datetime import datetime
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

def test_cosmos_connection():
    """Simple connection test function"""
    print("🚀 Starting Azure Cosmos DB Connection Test")
    print("=" * 50)
    
    try:
        # Get configuration from environment
        endpoint = os.environ.get('AZURE_COSMOSDB_ENDPOINT')
        database_name = os.environ.get('AZURE_COSMOSDB_DATABASE_NAME', 'testDatabase')
        container_name = os.environ.get('AZURE_COSMOSDB_CONTAINER_NAME', 'testContainer')
        
        if not endpoint:
            raise ValueError("❌ AZURE_COSMOSDB_ENDPOINT environment variable is required")
        
        print(f"📍 Endpoint: {endpoint}")
        print(f"🗄️  Database: {database_name}")
        print(f"📦 Container: {container_name}")
        
        # Connect using OIDC (DefaultAzureCredential)
        print("🔐 Authenticating with Azure...")
        credential = DefaultAzureCredential()
        client = CosmosClient(endpoint, credential)
        print("✅ Authentication successful")
        
        # Create database and container if they don't exist
        print("🔄 Creating database and container...")
        database = client.create_database_if_not_exists(id=database_name)
        container = database.create_container_if_not_exists(
            id=container_name,
            partition_key="/partitionKey"
        )
        print("✅ Database and container ready")
        
        # Test basic CRUD operations
        print("🧪 Testing basic operations...")
        
        # CREATE
        test_doc = {
            'id': str(uuid.uuid4()),
            'partitionKey': 'test',
            'type': 'connection_test',
            'message': 'Hello from GitHub Actions!',
            'timestamp': datetime.now().isoformat()
        }
        
        created_item = container.create_item(body=test_doc)
        print(f"✅ CREATE: Document created with ID: {created_item['id'][:8]}...")
        
        # READ
        read_item = container.read_item(
            item=created_item['id'], 
            partition_key='test'
        )
        print(f"✅ READ: Retrieved message: '{read_item['message']}'")
        
        # QUERY
        query = "SELECT * FROM c WHERE c.type = 'connection_test'"
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        print(f"✅ QUERY: Found {len(items)} test documents")
        
        # DELETE
        container.delete_item(
            item=created_item['id'], 
            partition_key='test'
        )
        print("✅ DELETE: Test document cleaned up")
        
        # Import sample data
        print("📥 Importing sample data...")
        sample_count = 0
        for i in range(1, 4):  # Just 3 samples
            sample_doc = {
                'id': str(uuid.uuid4()),
                'partitionKey': f'sample_{i}',
                'type': 'sample_data',
                'name': f'Sample Record {i}',
                'value': i * 10,
                'created_at': datetime.now().isoformat()
            }
            container.create_item(body=sample_doc)
            sample_count += 1
        
        print(f"✅ Imported {sample_count} sample records")
        
        print("=" * 50)
        print("🎉 All tests passed! Cosmos DB connection is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {str(e)}")
        print("=" * 50)
        return False

def main():
    """Main function to run the test"""
    success = test_cosmos_connection()
    
    if not success:
        exit(1)
    
    print("✅ Test completed successfully")

if __name__ == "__main__":
    main()