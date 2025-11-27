from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

SEARCH_ENDPOINT = "https://pen-match-api-v2-search.search.windows.net"
INDEX_NAME = "student-index"

credential = DefaultAzureCredential()

index_client = SearchIndexClient(
    endpoint=SEARCH_ENDPOINT,
    credential=credential
)

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=credential
)

# ---------------------------------------------------------------
# 1. CHECK INDEX SCHEMA (FIELDS, TYPES, FLAGS)
# ---------------------------------------------------------------
def print_index_schema():
    print("\n=== INDEX SCHEMA ===")
    index = index_client.get_index(INDEX_NAME)
    for f in index.fields:
        print(f"- {f.name:20} | {f.type:25} | searchable={f.searchable} filterable={f.filterable} sortable={f.sortable}")

# ---------------------------------------------------------------
# 2. COUNT TOTAL DOCUMENTS
# ---------------------------------------------------------------
def count_documents():
    print("\n=== COUNTING DOCUMENTS ===")
    count = search_client.get_document_count()
    print(f"Total documents: {count}")
    return count

# ---------------------------------------------------------------
# 3. READ SAMPLE DOCUMENTS (TOP 10)
# ---------------------------------------------------------------
def print_sample_docs():
    print("\n=== SAMPLE DOCUMENTS (TOP 10) ===")
    results = search_client.search(
        search="*",
        top=10
    )
    for doc in results:
        print(doc)
        print("--------------------------------------------------")

# ---------------------------------------------------------------
# 4. FETCH SPECIFIC DOCUMENT BY KEY
# ---------------------------------------------------------------
def get_by_id(doc_id):
    print(f"\n=== FETCH DOCUMENT id={doc_id} ===")
    try:
        doc = search_client.get_document(doc_id)
        print(doc)
    except Exception as e:
        print("Document not found:", e)

# ---------------------------------------------------------------
# 5. SHOW VECTOR FIELD SIZE (check embedding dimension)
# ---------------------------------------------------------------
def check_vector_length(n=5):
    print("\n=== VECTOR LENGTH CHECK ===")
    results = search_client.search(search="*", top=n, select=["id", "nameEmbedding"])
    for r in results:
        emb = r.get("nameEmbedding", None)
        if emb:
            print(f"id={r['id']} | embedding length={len(emb)}")
        else:
            print(f"id={r['id']} | NO embedding found")

# ---------------------------------------------------------------
# 6. UPDATE DOCUMENT (MERGE / OVERWRITE)
# ---------------------------------------------------------------
def update_document(doc):
    print("\n=== UPDATING DOCUMENT ===")
    # MERGE OR CREATE (upsert)
    result = search_client.merge_or_upload_documents([doc])
    print("Update result:", result)

# ---------------------------------------------------------------
# RUN EVERYTHING
# ---------------------------------------------------------------
if __name__ == "__main__":
    print_index_schema()
    count_documents()
    print_sample_docs()
    check_vector_length()
