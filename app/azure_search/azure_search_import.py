import asyncio
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
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
    def __init__(self, max_concurrent_batches=10, max_db_connections=20, thread_pool_size=20):
        self.search_endpoint = "https://pen-match-api-v2-search.search.windows.net"
        self.index_name = "student-index"
        
        # Azure Search client
        self.credential = DefaultAzureCredential()
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
        
        # OpenAI embedding client - use AzureOpenAI like student_embedding.py
        self.openai_client = AzureOpenAI(
            api_key=settings.openai_api_key,
            api_version="2023-05-15",
            azure_endpoint=settings.openai_api_base_embedding_3
        )
        
        # Processing setup
        self.max_concurrent_batches = max_concurrent_batches
        self.thread_pool_size = thread_pool_size
        self.db = PostgreSQLManager(max_db_connections)
        self.stats = AzureSearchProcessingStats()
        
        print("Initializing AzureSearchImportService...")
        print(f"Embedding endpoint: {settings.openai_api_base_embedding_3}")
        print("AzureSearchImportService initialized successfully")
    
    def generate_embedding(self, student: Dict[str, Any]) -> List[float]:
        """Generate embedding using text-embedding-3-large model"""
        # Format: [name: first name last name, middlename: middle name]
        first_name = student.get('legalFirstName', '').strip()
        last_name = student.get('legalLastName', '').strip()
        middle_names = student.get('legalMiddleNames', '').strip()
        
        if first_name == 'NULL':
            first_name = ''
        if last_name == 'NULL':
            last_name = ''
        if middle_names == 'NULL':
            middle_names = ''
        
        name_part = f"{first_name} {last_name}".strip()
        text = f"[name: {name_part}, middlename: {middle_names}]"
        
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error for student {student.get('student_id')}: {e}")
            raise
    
    def _prepare_search_document(self, student: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
        """Prepare student data for Azure Search index"""
        content_parts = []
        if student.get('legalFirstName') and student.get('legalFirstName') != 'NULL':
            content_parts.append(student['legalFirstName'])
        if student.get('legalMiddleNames') and student.get('legalMiddleNames') != 'NULL':
            content_parts.append(student['legalMiddleNames'])
        if student.get('legalLastName') and student.get('legalLastName') != 'NULL':
            content_parts.append(student['legalLastName'])
        if student.get('pen') and student.get('pen') != 'NULL':
            content_parts.append(student['pen'])
        
        content = ' '.join(content_parts)
        
        return {
            'id': student['student_id'],
            'student_id': student['student_id'],
            'pen': student.get('pen') if student.get('pen') != 'NULL' else None,
            'legalFirstName': student.get('legalFirstName') if student.get('legalFirstName') != 'NULL' else None,
            'legalMiddleNames': student.get('legalMiddleNames') if student.get('legalMiddleNames') != 'NULL' else None,
            'legalLastName': student.get('legalLastName') if student.get('legalLastName') != 'NULL' else None,
            'content': content,
            'nameEmbedding': embedding,
            'dob': student.get('dob') if student.get('dob') != 'NULL' else None,
            'sexCode': student.get('sexCode') if student.get('sexCode') != 'NULL' else None,
            'postalCode': student.get('postalCode') if student.get('postalCode') != 'NULL' else None,
            'mincode': student.get('mincode') if student.get('mincode') != 'NULL' else None,
            'grade': student.get('grade') if student.get('grade') != 'NULL' else None,
            'localID': student.get('localID') if student.get('localID') != 'NULL' else None
        }
    
    async def _save_embedding_to_db(self, student_id: str, embedding: List[float]):
        """Save embedding to PostgreSQL"""
        async with self.db.connection_pool.acquire() as conn:
            query = '''
                UPDATE "api_pen_match_v2".student 
                SET embedding = $1::float[]
                WHERE student_id = $2
            '''
            await conn.execute(query, embedding, student_id)
    
    def _generate_search_documents_batch(self, students: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate search documents for a batch of students"""
        results = []
        for student in students:
            try:
                embedding = self.generate_embedding(student)
                document = self._prepare_search_document(student, embedding)
                
                results.append({
                    'document': document,
                    'student_id': student['student_id'],
                    'embedding': embedding,
                    'success': True
                })
                
            except Exception as e:
                print(f"Error processing student {student.get('student_id')}: {e}")
                results.append({
                    'document': None,
                    'student_id': student['student_id'],
                    'embedding': None,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def _batch_upload_and_save(self, results: List[Dict[str, Any]]) -> int:
        """Upload to Azure Search and save embeddings to PostgreSQL"""
        successful_results = [r for r in results if r['success'] and r['document']]
        if not successful_results:
            return 0
        
        documents = [r['document'] for r in successful_results]
        
        try:
            # Upload to Azure Search
            upload_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.search_client.upload_documents(documents)
            )
            
            uploaded_count = 0
            for result in upload_result:
                if result.succeeded:
                    uploaded_count += 1
                    
                    # Save embedding to PostgreSQL
                    matching_result = next(r for r in successful_results if r['document']['id'] == result.key)
                    await self._save_embedding_to_db(matching_result['student_id'], matching_result['embedding'])
                else:
                    print(f"Failed to upload document {result.key}: {result.error_message}")
            
            return uploaded_count
            
        except Exception as e:
            print(f"Error in batch upload: {e}")
            return 0
    
    async def import_one_batch(self, offset=0, batch_size=100):
        """Import single batch to Azure Search and save embeddings"""
        print(f"Starting import for batch at offset {offset} with batch size {batch_size}")
        
        await self.db.create_pool()
        try:
            students = await self.db.fetch_students_batch(offset, batch_size)
            if not students:
                print(f"No students found at offset {offset}")
                return 0
            
            uploaded = 0
            for student in students:
                try:
                    student_id = student.get("student_id")
                    print(f"Processing student: {student_id}")
                    
                    embedding = self.generate_embedding(student)
                    document = self._prepare_search_document(student, embedding)
                    
                    # Upload to Azure Search
                    result = self.search_client.upload_documents([document])
                    if result[0].succeeded:
                        # Save embedding to PostgreSQL
                        await self._save_embedding_to_db(student_id, embedding)
                        uploaded += 1
                        print(f"Successfully processed: {student_id}")
                    else:
                        print(f"Failed to upload: {student_id}")
                        
                except Exception as e:
                    print(f"Error processing student {student.get('student_id')}: {e}")
                    continue
            
            print(f"Batch completed: {uploaded}/{len(students)} processed")
            return uploaded
        
        finally:
            await self.db.close()

    async def import_all_students(self, batch_size=1000):
        """Import all students with parallel processing"""
        print(f"Starting import of all students (batch_size: {batch_size})")
        
        self.stats.start_time = time.time()
        await self.db.create_pool()
        
        try:
            total_count = await self.db.get_total_student_count()
            print(f"Total records: {total_count:,}")
            
            with ThreadPoolExecutor(max_workers=self.thread_pool_size) as executor:
                for offset in range(0, total_count, batch_size):
                    students = await self.db.fetch_students_batch(offset, batch_size)
                    if not students:
                        break
                    
                    results = await self._process_students_parallel(students, executor)
                    uploaded = await self._batch_upload_and_save(results)
                    
                    self.stats.total_processed += uploaded
                    self.stats.total_failed += len(students) - uploaded
                    self.stats.batches_completed += 1
                    
                    if self.stats.batches_completed % 10 == 0:
                        elapsed = time.time() - self.stats.start_time
                        rate = self.stats.total_processed / elapsed if elapsed > 0 else 0
                        print(f"Progress: {self.stats.total_processed:,} processed ({rate:.1f}/sec)")
            
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
                        SELECT student_id, 
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
                    
                    uploaded_for_name = 0
                    
                    for j, row in enumerate(rows, 1):
                        try:
                            student_id = row["student_id"]
                            print(f"  Processing record {j}/{len(rows)} - Student ID: {student_id}")
                            
                            # Create student object with correct field mapping
                            student = {
                                "student_id": student_id,
                                "pen": row["pen"],
                                "legalFirstName": row["legal_first_name"],
                                "legalLastName": row["legal_last_name"],
                                "legalMiddleNames": row["legal_middle_names"],
                                "dob": row["dob"],
                                "sexCode": row["sex_code"],
                                "postalCode": row["postal_code"],
                                "mincode": row["mincode"],
                                "grade": row["grade_code"],
                                "localID": row["local_id"]
                            }
                            
                            # Generate embedding and document
                            embedding = self.generate_embedding(student)
                            document = self._prepare_search_document(student, embedding)
                            
                            # Upload to Azure Search
                            result = self.search_client.upload_documents([document])
                            if result[0].succeeded:
                                # Save embedding to PostgreSQL
                                await self._save_embedding_to_db(student_id, embedding)
                                uploaded_for_name += 1
                                print(f"    Successfully processed: {student_id}")
                            else:
                                print(f"    Failed to upload: {student_id}")
                            
                        except Exception as e:
                            print(f"    Error processing student {row.get('student_id')}: {e}")
                            continue
                    
                    total_uploaded += uploaded_for_name
                    print(f"Completed {first_name} {last_name}: {uploaded_for_name} uploaded")
            
            print(f"\nName-based import completed: {total_uploaded} total students processed")
            return total_uploaded
            
        finally:
            await self.db.close()

    async def _process_students_parallel(self, students: List[Dict[str, Any]], executor: ThreadPoolExecutor) -> List[Dict[str, Any]]:
        """Process students in parallel chunks"""
        if not students:
            return []
        
        chunk_size = max(1, len(students) // self.thread_pool_size)
        chunks = [students[i:i + chunk_size] for i in range(0, len(students), chunk_size)]
        
        loop = asyncio.get_event_loop()
        futures = [loop.run_in_executor(executor, self._generate_search_documents_batch, chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*futures, return_exceptions=True)
        
        results = []
        for chunk_result in chunk_results:
            if isinstance(chunk_result, list):
                results.extend(chunk_result)
        
        return results


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