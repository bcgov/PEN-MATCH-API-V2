import json
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
SEARCH_SERVICE_ENDPOINT = "https://pen-match-api-v2-search.search.windows.net"
INDEX_JSON_PATH = "azure_search/azure_search_index.json"   # <- your schema file
INDEX_NAME = "student-index"

# -----------------------------------------------------------
# AUTHENTICATION
# -----------------------------------------------------------
# Uses VM’s Managed Identity automatically
credential = DefaultAzureCredential()

# Create index client
index_client = SearchIndexClient(
    endpoint=SEARCH_SERVICE_ENDPOINT,
    credential=credential
)

# -----------------------------------------------------------
# LOAD INDEX SCHEMA JSON
# -----------------------------------------------------------
with open(INDEX_JSON_PATH, "r") as f:
    index_schema = json.load(f)

# Deserialize JSON into SearchIndex model
index_object = SearchIndex.deserialize(index_schema)

# -----------------------------------------------------------
# CHECK & DELETE EXISTING INDEX (optional but recommended)
# -----------------------------------------------------------
try:
    print(f"Checking if index '{INDEX_NAME}' exists...")
    existing = index_client.get_index(INDEX_NAME)
    print(f"Index '{INDEX_NAME}' already exists → deleting it...")
    index_client.delete_index(INDEX_NAME)
    print("Old index deleted.")
except Exception:
    print("Index does not exist — creating new one.")

# -----------------------------------------------------------
# CREATE INDEX
# -----------------------------------------------------------
print(f"Creating index '{INDEX_NAME}'...")
index_client.create_index(index_object)
print("Index created successfully!")
