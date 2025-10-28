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
    container_name = "StudentAPITest"
    credential = DefaultAzureCredential()

    try:
        client = CosmosClient(endpoint, credential)
        database = client.get_database_client(database_name)
        container = database.get_container_client(container_name)

        # Insert the document
        container.upsert_item(sample_document)
        logger.info("✅ Document inserted successfully.")
        return container
    except Exception as e:
        logger.error(f"❌ Failed to insert document: {str(e)}")
        return None

def query_documents(container):
    try:
        query = "SELECT * FROM c WHERE c.pen = 350355"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        logger.info(f"✅ Retrieved {len(items)} documents with pen = 350355")
        for item in items:
            print(item)
        return items
    except Exception as e:
        logger.error(f"❌ Failed to query documents: {str(e)}")
        return []

if __name__ == "__main__":
    logger.info("=== COSMOS DB INSERT TEST START ===")
    container = insert_document()
    logger.info("=== COSMOS DB INSERT TEST END ===")

    if container:
        logger.info("=== COSMOS DB QUERY TEST START ===")
        query_documents(container)
        logger.info("=== COSMOS DB QUERY TEST END ===")