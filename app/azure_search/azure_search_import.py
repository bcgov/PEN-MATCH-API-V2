import asyncio
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.core.exceptions import HttpResponseError
from openai import AzureOpenAI

from database.postgresql import PostgreSQLManager
from config.settings import settings

@dataclass
class AzureSearchProcessingStats:
    total_processed: int = 0
    total_failed: int = 0
    start_time: float = 0
    batches_completed: int = 0

class AzureSearchImportService:
    def __init__(self, max_db_connections=20):
        self.search_endpoint = "https://pen-match-api-v2-search.search.windows.net"
        self.index_name = "student-index"
        
        # Azure Search client
        self.credential = DefaultAzureCredential()
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
        
        # OpenAI embedding client
        self.openai_client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version="2023-05-15",
            azure_endpoint=settings.openai_api_base_embedding_3
        )
        
        # Database setup
        self.db = PostgreSQLManager(max_db_connections)
        self.stats = AzureSearchProcessingStats()
        
        print("Initializing AzureSearchImportService...")
        print(f"Embedding endpoint: {settings.openai_api_base_embedding_3}")
        print("AzureSearchImportService initialized successfully")
    
    def generate_embedding(self, student: Dict[str, Any]) -> List[float]:
        """Generate embedding using text-embedding-3-large model"""
        # Format: Name: FIRST MIDDLE LAST.
        first_name = student.get('legalFirstName', '').strip()
        last_name = student.get('legalLastName', '').strip()
        middle_names = student.get('legalMiddleNames', '').strip()
        
        if first_name == 'NULL':
            first_name = ''
        if last_name == 'NULL':
            last_name = ''
        if middle_names == 'NULL':
            middle_names = ''
        
        # Build full name with middle names in between
        name_parts = []
        if first_name:
            name_parts.append(first_name)
        if middle_names:
            name_parts.append(middle_names)
        if last_name:
            name_parts.append(last_name)
        
        full_name = ' '.join(name_parts)
        text = f"{full_name}."
        
        print(f"Generating embedding for: {text}")
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                dimensions=3072
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"Error generating embedding for '{text}': {e}")
            # Return zero vector as fallback
            return [0.0] * 3072
    
    def _prepare_search_document(self, student: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare student data for Azure Search index"""
        # Generate embedding first
        embedding = self.generate_embedding(student)
        
        # Build comprehensive content field
        content_parts = []
        
        # Name
        name_parts = []
        if student.get('legalFirstName') and student.get('legalFirstName') != 'NULL':
            name_parts.append(student['legalFirstName'])
        if student.get('legalMiddleNames') and student.get('legalMiddleNames') != 'NULL':
            name_parts.append(student['legalMiddleNames'])
        if student.get('legalLastName') and student.get('legalLastName') != 'NULL':
            name_parts.append(student['legalLastName'])
        
        if name_parts:
            content_parts.append(' '.join(name_parts))
        
        # Date of birth
        if student.get('dob') and student.get('dob') != 'NULL':
            content_parts.append(f"born {student['dob']}")
        
        # Sex
        if student.get('sexCode') and student.get('sexCode') != 'NULL':
            sex_text = "male" if student['sexCode'].upper() == 'M' else "female"
            content_parts.append(sex_text)
        
        # Postal code
        if student.get('postalCode') and student.get('postalCode') != 'NULL':
            content_parts.append(f"postal {student['postalCode']}")
        
        # School code (mincode)
        if student.get('mincode') and student.get('mincode') != 'NULL':
            content_parts.append(f"mincode {student['mincode']}")
        
        content = ', '.join(content_parts)
        
        return {
            'id': str(student['student_id']),
            'student_id': str(student['student_id']),
            'pen': student.get('pen') if student.get('pen') != 'NULL' else None,
            'legalFirstName': student.get('legalFirstName') if student.get('legalFirstName') != 'NULL' else None,
            'legalMiddleNames': student.get('legalMiddleNames') if student.get('legalMiddleNames') != 'NULL' else None,
            'legalLastName': student.get('legalLastName') if student.get('legalLastName') != 'NULL' else None,
            'content': content,
            'nameEmbedding': embedding,  # Add the generated embedding
            'dob': student.get('dob') if student.get('dob') != 'NULL' else None,
            'sexCode': student.get('sexCode') if student.get('sexCode') != 'NULL' else None,
            'postalCode': student.get('postalCode') if student.get('postalCode') != 'NULL' else None,
            'mincode': student.get('mincode') if student.get('mincode') != 'NULL' else None,
            'gradeCode': student.get('gradeCode') if student.get('gradeCode') != 'NULL' else None,
            'localID': student.get('localID') if student.get('localID') != 'NULL' else None
        }
    
    async def _batch_upload(self, documents: List[Dict[str, Any]]) -> int:
        """Upload batch of documents to Azure Search"""
        if not documents:
            return 0
        
        try:
            upload_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.search_client.upload_documents(documents)
            )
            
            uploaded_count = 0
            for result in upload_result:
                if result.succeeded:
                    uploaded_count += 1
                else:
                    print(f"Failed to upload document {result.key}: {result.error_message}")
            
            return uploaded_count
            
        except Exception as e:
            print(f"Error in batch upload: {e}")
            return 0
    
    async def import_one_batch(self, offset=0, batch_size=100):
        """Import single batch to Azure Search"""
        print(f"Starting import for batch at offset {offset} with batch size {batch_size}")
        
        await self.db.create_pool()
        try:
            async with self.db.connection_pool.acquire() as conn:
                query = """
                    SELECT student_id::text as student_id, 
                           COALESCE(pen, 'NULL') as pen, 
                           COALESCE(legal_first_name, 'NULL') as legal_first_name,
                           COALESCE(legal_last_name, 'NULL') as legal_last_name, 
                           COALESCE(legal_middle_names, 'NULL') as legal_middle_names,
                           COALESCE(dob::text, 'NULL') as dob, 
                           COALESCE(sex_code, 'NULL') as sex_code,
                           COALESCE(postal_code, 'NULL') as postal_code, 
                           COALESCE(mincode, 'NULL') as mincode,
                           COALESCE(grade_code, 'NULL') as grade_code,
                           COALESCE(LPAD(local_id::text, 8, '0'), 'NULL') as local_id
                    FROM "api_pen_match_v2".student 
                    ORDER BY student_id ASC
                    LIMIT $1 OFFSET $2
                """
                
                rows = await conn.fetch(query, batch_size, offset)
                
                if not rows:
                    print(f"No students found at offset {offset}")
                    return 0
                
                documents = []
                for i, row in enumerate(rows):
                    try:
                        # Create student object with correct field mapping
                        student = {
                            "student_id": str(row["student_id"]),
                            "pen": row["pen"],
                            "legalFirstName": row["legal_first_name"],
                            "legalLastName": row["legal_last_name"],
                            "legalMiddleNames": row["legal_middle_names"],
                            "dob": row["dob"],
                            "sexCode": row["sex_code"],
                            "postalCode": row["postal_code"],
                            "mincode": row["mincode"],
                            "gradeCode": row["grade_code"],
                            "localID": row["local_id"]
                        }
                        
                        print(f"Processing student {i+1}/{len(rows)}: {student.get('legalFirstName', '')} {student.get('legalLastName', '')}")
                        document = self._prepare_search_document(student)
                        documents.append(document)
                        
                    except Exception as e:
                        print(f"Error preparing document for student {row.get('student_id')}: {e}")
                        continue
                
                uploaded = await self._batch_upload(documents)
                print(f"Batch completed: {uploaded}/{len(rows)} processed")
                return uploaded
        
        finally:
            await self.db.close()

    async def import_all_students(self, batch_size=1000):
        """Import all students in batches"""
        print(f"Starting import of all students (batch_size: {batch_size})")
        
        self.stats.start_time = time.time()
        await self.db.create_pool()
        
        try:
            # Get total count
            async with self.db.connection_pool.acquire() as conn:
                count_result = await conn.fetchval('SELECT COUNT(*) FROM "api_pen_match_v2".student')
                total_count = count_result or 0
            
            print(f"Total records: {total_count:,}")
            
            for offset in range(0, total_count, batch_size):
                print(f"Processing batch: {offset} to {offset + batch_size}")
                
                async with self.db.connection_pool.acquire() as conn:
                    query = """
                        SELECT student_id::text as student_id, 
                               COALESCE(pen, 'NULL') as pen, 
                               COALESCE(legal_first_name, 'NULL') as legal_first_name,
                               COALESCE(legal_last_name, 'NULL') as legal_last_name, 
                               COALESCE(legal_middle_names, 'NULL') as legal_middle_names,
                               COALESCE(dob::text, 'NULL') as dob, 
                               COALESCE(sex_code, 'NULL') as sex_code,
                               COALESCE(postal_code, 'NULL') as postal_code, 
                               COALESCE(mincode, 'NULL') as mincode,
                               COALESCE(grade_code, 'NULL') as grade_code,
                               COALESCE(LPAD(local_id::text, 8, '0'), 'NULL') as local_id
                        FROM "api_pen_match_v2".student 
                        ORDER BY student_id ASC
                        LIMIT $1 OFFSET $2
                    """
                    
                    rows = await conn.fetch(query, batch_size, offset)
                    if not rows:
                        break
                    
                    documents = []
                    for i, row in enumerate(rows):
                        try:
                            # Create student object with correct field mapping
                            student = {
                                "student_id": str(row["student_id"]),
                                "pen": row["pen"],
                                "legalFirstName": row["legal_first_name"],
                                "legalLastName": row["legal_last_name"],
                                "legalMiddleNames": row["legal_middle_names"],
                                "dob": row["dob"],
                                "sexCode": row["sex_code"],
                                "postalCode": row["postal_code"],
                                "mincode": row["mincode"],
                                "gradeCode": row["grade_code"],
                                "localID": row["local_id"]
                            }
                            
                            if (i + 1) % 50 == 0:  # Progress indicator for large batches
                                print(f"Processing student {i+1}/{len(rows)} in current batch")
                            
                            document = self._prepare_search_document(student)
                            documents.append(document)
                            
                        except Exception as e:
                            print(f"Error preparing document for student {row.get('student_id')}: {e}")
                            continue
                    
                    uploaded = await self._batch_upload(documents)
                    
                    self.stats.total_processed += uploaded
                    self.stats.total_failed += len(rows) - uploaded
                    self.stats.batches_completed += 1
                    
                    elapsed = time.time() - self.stats.start_time
                    rate = self.stats.total_processed / elapsed if elapsed > 0 else 0
                    print(f"Batch {self.stats.batches_completed}: {uploaded} uploaded, Total: {self.stats.total_processed:,} ({rate:.1f}/sec)")
            
            elapsed = time.time() - self.stats.start_time
            print(f"Import completed: {self.stats.total_processed:,} students in {elapsed:.1f}s")
            return self.stats.total_processed
            
        finally:
            await self.db.close()

    async def import_all_records_by_names(self, target_names: List[tuple]) -> int:
        """Import all student records that match specified name pairs"""
        print(f"Starting import for {len(target_names)} name pairs")
        
        await self.db.create_pool()
        try:
            total_uploaded = 0
            
            for i, (first_name, last_name) in enumerate(target_names, 1):
                print(f"\nProcessing name pair {i}/{len(target_names)}: {first_name} {last_name}")
                
                async with self.db.connection_pool.acquire() as conn:
                    query = """
                        SELECT student_id::text as student_id, 
                               COALESCE(pen, 'NULL') as pen, 
                               COALESCE(legal_first_name, 'NULL') as legal_first_name,
                               COALESCE(legal_last_name, 'NULL') as legal_last_name, 
                               COALESCE(legal_middle_names, 'NULL') as legal_middle_names,
                               COALESCE(dob::text, 'NULL') as dob, 
                               COALESCE(sex_code, 'NULL') as sex_code,
                               COALESCE(postal_code, 'NULL') as postal_code, 
                               COALESCE(mincode, 'NULL') as mincode,
                               COALESCE(grade_code, 'NULL') as grade_code,
                               COALESCE(LPAD(local_id::text, 8, '0'), 'NULL') as local_id
                        FROM "api_pen_match_v2".student 
                        WHERE LOWER(TRIM(legal_first_name)) = LOWER($1) 
                        AND LOWER(TRIM(legal_last_name)) = LOWER($2)
                        ORDER BY student_id ASC
                    """
                    
                    rows = await conn.fetch(query, first_name.strip(), last_name.strip())
                    print(f"Found {len(rows)} records for {first_name} {last_name}")
                    
                    if not rows:
                        continue
                    
                    documents = []
                    for j, row in enumerate(rows):
                        try:
                            # Create student object with correct field mapping
                            student = {
                                "student_id": str(row["student_id"]),
                                "pen": row["pen"],
                                "legalFirstName": row["legal_first_name"],
                                "legalLastName": row["legal_last_name"],
                                "legalMiddleNames": row["legal_middle_names"],
                                "dob": row["dob"],
                                "sexCode": row["sex_code"],
                                "postalCode": row["postal_code"],
                                "mincode": row["mincode"],
                                "gradeCode": row["grade_code"],
                                "localID": row["local_id"]
                            }
                            
                            print(f"  Processing record {j+1}/{len(rows)}: {student.get('legalFirstName', '')} {student.get('legalMiddleNames', '')} {student.get('legalLastName', '')}")
                            document = self._prepare_search_document(student)
                            documents.append(document)
                            
                        except Exception as e:
                            print(f"Error preparing document for student {row.get('student_id')}: {e}")
                            continue
                    
                    uploaded_for_name = await self._batch_upload(documents)
                    total_uploaded += uploaded_for_name
                    print(f"Completed {first_name} {last_name}: {uploaded_for_name} uploaded")
            
            print(f"\nName-based import completed: {total_uploaded} total students processed")
            return total_uploaded
            
        finally:
            await self.db.close()

# Standalone functions for compatibility
async def import_student_data():
    """Import sample student data for testing"""
    service = AzureSearchImportService()
    return await service.import_one_batch(0, 1)

def import_one_batch(offset=0, batch_size=100):
    """Synchronous wrapper for batch import"""
    service = AzureSearchImportService()
    return asyncio.run(service.import_one_batch(offset, batch_size))

def import_all_students(batch_size=1000):
    """Synchronous wrapper for importing all students"""
    service = AzureSearchImportService()
    return asyncio.run(service.import_all_students(batch_size))

def import_by_names(target_names: List[tuple]):
    """Synchronous wrapper for importing by names"""
    service = AzureSearchImportService()
    return asyncio.run(service.import_all_records_by_names(target_names))

if __name__ == "__main__":
    import sys
    
    async def main():
        service = AzureSearchImportService()
        
        try:
            if len(sys.argv) > 1:
                if sys.argv[1] == "all":
                    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
                    result = await service.import_all_students(batch_size)
                elif sys.argv[1] == "names":
                    target_names = [
                        ("ROBYN", "ANDERSON"),
                        ("MICHAEL", "LEE"),
                        ("MICHAEL", "KIM"),
                        ("DAVID", "LEE"),
                        ("MICHAEL", "WANG"),
                        ("JENNIFER", "LEE"),
                        ("MICHAEL", "LI"),
                        ("ROBERT", "LEE"),
                        ("DAVID", "WANG"),
                        ("MICHAEL", "CHEN"),
                        ("JAMES", "LEE")
                    ]
                    result = await service.import_all_records_by_names(target_names)
                else:
                    offset = int(sys.argv[1])
                    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
                    result = await service.import_one_batch(offset, batch_size)
            else:
                result = await service.import_one_batch()
            
            print(f"Final result: {result} students processed")
            return 0
                
        except Exception as e:
            print(f"Import failed: {e}")
            return 1
    
    exit_code = asyncio.run(main())
    exit(exit_code)