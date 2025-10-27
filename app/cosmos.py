from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
import os

def test_connection():
    endpoint = os.environ.get('AZURE_COSMOSDB_ENDPOINT')
    credential = DefaultAzureCredential()
    
    try:
        client = CosmosClient(endpoint, credential)
        databases = list(client.list_databases())
        return f"✅ Connected! Found {len(databases)} databases"
    except Exception as e:
        return f"❌ Failed: {str(e)}"

if __name__ == "__main__":
    print(test_connection())