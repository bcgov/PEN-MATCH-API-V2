from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import DefaultAzureCredential
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample data to insert
sample_document = {
    "id": "pen-350355",  # Unique ID for Cosmos DB
    "pen": 350355,
    "legalFirstName": "RYN",
    "legalMiddleNames": "DELLY",
    "legalLastName": "ASON",
    "dob": "2010-11-09",
    "localID": "8929223"
}

def insert_document():
    endpoint = "https://pen-match-api-v2-cosmos.documents.azure.com:443/"
    database_name = "PenMatchDB"
    container_name = "StudentAPITest"  # e.g., "Students"
    credential = DefaultAzureCredential()

    try:
        client = CosmosClient(endpoint, credential)
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)

        # Insert the document
        container.upsert_item(sample_document)
        logger.info("✅ Document inserted successfully.")
        return "✅ Document inserted successfully."
    except Exception as e:
        logger.error(f"❌ Failed to insert document: {str(e)}")
        return f"❌ Failed to insert document: {str(e)}"

if __name__ == "__main__":
    logger.info("=== COSMOS DB INSERT TEST START ===")
    result = insert_document()
    print(result)
    logger.info("=== COSMOS DB INSERT TEST END ===")