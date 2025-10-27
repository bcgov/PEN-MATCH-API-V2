from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
import os
import logging

# Configure logging for App Service
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_connection():
    endpoint = os.environ.get('AZURE_COSMOSDB_ENDPOINT')
    credential = DefaultAzureCredential()
    
    try:
        client = CosmosClient(endpoint, credential)
        databases = list(client.list_databases())
        result = f"✅ Connected! Found {len(databases)} databases"
        logger.info(f"COSMOS_TEST_RESULT: {result}")  # Special log marker
        return result
    except Exception as e:
        result = f"❌ Failed: {str(e)}"
        logger.error(f"COSMOS_TEST_RESULT: {result}")  # Special log marker
        return result

if __name__ == "__main__":
    logger.info("=== COSMOS DB CONNECTION TEST START ===")
    result = test_connection()
    print(result)
    logger.info("=== COSMOS DB CONNECTION TEST END ===")