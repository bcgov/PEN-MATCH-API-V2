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
    try:
        index = index_client.get_index(INDEX_NAME)
        for f in index.fields:
            print(f"- {f.name:20} | {f.type:25} | searchable={f.searchable} filterable={f.filterable} sortable={f.sortable}")
    except Exception as e:
        print(f"Error getting index schema: {e}")

# ---------------------------------------------------------------
# 2. COUNT TOTAL DOCUMENTS
# ---------------------------------------------------------------
def count_documents():
    print("\n=== COUNTING DOCUMENTS ===")
    try:
        count = search_client.get_document_count()
        print(f"Total documents: {count}")
        return count
    except Exception as e:
        print(f"Error counting documents: {e}")
        return 0

# ---------------------------------------------------------------
# 3. READ SAMPLE DOCUMENTS (TOP 10) - FIXED
# ---------------------------------------------------------------
def print_sample_docs():
    print("\n=== SAMPLE DOCUMENTS (TOP 10) ===")
    try:
        # Use search_text parameter instead of search
        results = search_client.search(
            search_text="*",
            top=10,
            include_total_count=True
        )
        
        count = 0
        for doc in results:
            count += 1
            print(f"Document {count}:")
            # Print key fields only to avoid clutter
            for key, value in doc.items():
                if key == 'nameEmbedding':
                    # Show embedding info without printing the full array
                    if value:
                        print(f"  {key}: [embedding with {len(value)} dimensions]")
                    else:
                        print(f"  {key}: None")
                else:
                    print(f"  {key}: {value}")
            print("--------------------------------------------------")
            
        if count == 0:
            print("No documents found in the index")
            
    except Exception as e:
        print(f"Error searching documents: {e}")

# ---------------------------------------------------------------
# 4. FETCH SPECIFIC DOCUMENT BY KEY
# ---------------------------------------------------------------
def get_by_id(doc_id):
    print(f"\n=== FETCH DOCUMENT student_id={doc_id} ===")
    try:
        doc = search_client.get_document(doc_id)
        for key, value in doc.items():
            if key == 'nameEmbedding':
                if value:
                    print(f"  {key}: [embedding with {len(value)} dimensions]")
                else:
                    print(f"  {key}: None")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print("Document not found:", e)

# ---------------------------------------------------------------
# 5. SHOW VECTOR FIELD SIZE (check embedding dimension)
# ---------------------------------------------------------------
def check_vector_length(n=5):
    print("\n=== VECTOR LENGTH CHECK ===")
    try:
        results = search_client.search(
            search_text="*", 
            top=n, 
            select=["student_id", "nameEmbedding"]
        )
        
        count = 0
        for r in results:
            count += 1
            emb = r.get("nameEmbedding", None)
            if emb:
                print(f"student_id={r['student_id']} | embedding length={len(emb)}")
            else:
                print(f"student_id={r['student_id']} | NO embedding found")
                
        if count == 0:
            print("No documents found for vector check")
            
    except Exception as e:
        print(f"Error checking vector length: {e}")

# ---------------------------------------------------------------
# 6. SEARCH BY NAME
# ---------------------------------------------------------------
def search_by_name(name, top=5):
    print(f"\n=== SEARCH BY NAME: {name} ===")
    try:
        results = search_client.search(
            search_text=name,
            top=top,
            select=["id", "legalFirstName", "legalLastName", "legalMiddleNames", "pen", "dob"]
        )
        
        count = 0
        for doc in results:
            count += 1
            print(f"Result {count}:")
            for key, value in doc.items():
                print(f"  {key}: {value}")
            print("--------------------------------------------------")
            
        if count == 0:
            print(f"No results found for '{name}'")
            
    except Exception as e:
        print(f"Error searching by name: {e}")

# ---------------------------------------------------------------
# 7. SEARCH BY PEN NUMBER
# ---------------------------------------------------------------
def search_pen(pen_number):
    print(f"\n=== SEARCH BY PEN: {pen_number} ===")
    try:
        results = search_client.search(
            search_text="*",
            filter=f"pen eq '{pen_number}'",
            top=10  # Should only be 1 result per PEN, but keep some buffer
        )
        
        count = 0
        for doc in results:
            count += 1
            print(f"Student {count} (PEN: {pen_number}):")
            for key, value in doc.items():
                if key == 'nameEmbedding':
                    # Skip nameEmbedding as requested
                    continue
                print(f"  {key}: {value}")
            print("--------------------------------------------------")
            
        if count == 0:
            print(f"No student found with PEN: {pen_number}")
        else:
            print(f"Found {count} student(s) with PEN: {pen_number}")
            
    except Exception as e:
        print(f"Error searching by PEN: {e}")

# ---------------------------------------------------------------
# 8. UPDATE DOCUMENT (MERGE / OVERWRITE)
# ---------------------------------------------------------------
def update_document(doc):
    print("\n=== UPDATING DOCUMENT ===")
    try:
        # MERGE OR CREATE (upsert)
        result = search_client.merge_or_upload_documents([doc])
        print("Update result:", result)
    except Exception as e:
        print(f"Error updating document: {e}")

# ---------------------------------------------------------------
# RUN EVERYTHING
# ---------------------------------------------------------------
if __name__ == "__main__":
    try:
        print_index_schema()
        count_documents()
        search_pen("")
        # print_sample_docs()
        # check_vector_length()
        # search_robyn_anderson()
        
        # Test with a specific PEN number
        # search_pen("124809765")
        
        # Test with a specific document ID if you have one
        # get_by_id("some-student-id-here")
        
    except Exception as e:
        print(f"General error: {e}")