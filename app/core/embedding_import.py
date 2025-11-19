import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from dataclasses import dataclass
from core.student_embedding import StudentEmbedding
from database.postgresql import PostgreSQLManager

@dataclass
class ProcessingStats:
    total_processed: int = 0
    total_failed: int = 0
    start_time: float = 0
    batches_completed: int = 0

class EmbeddingImportService:
    def __init__(self, max_concurrent_batches=10, max_db_connections=20, thread_pool_size=20):
        self.student_embedding = StudentEmbedding()
        self.max_concurrent_batches = max_concurrent_batches
        self.thread_pool_size = thread_pool_size
        self.db = PostgreSQLManager(max_db_connections)
        self.stats = ProcessingStats()
        print("Initializing EmbeddingImportService...")
        print("EmbeddingImportService initialized successfully")
    
    def _generate_embeddings_batch(self, students: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for a batch of students"""
        results = []
        for student in students:
            try:
                embedding = self.student_embedding.generate_embedding(student)
                results.append({'student_id': student['student_id'], 'embedding': embedding, 'success': True})
            except Exception:
                results.append({'student_id': student['student_id'], 'embedding': None, 'success': False})
        return results
    
    async def _process_students_parallel(self, students: List[Dict[str, Any]], executor: ThreadPoolExecutor) -> List[Dict[str, Any]]:
        """Process embeddings in parallel with chunking"""
        if not students:
            return []
        
        chunk_size = max(1, len(students) // self.thread_pool_size)
        chunks = [students[i:i + chunk_size] for i in range(0, len(students), chunk_size)]
        
        loop = asyncio.get_event_loop()
        futures = [loop.run_in_executor(executor, self._generate_embeddings_batch, chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*futures, return_exceptions=True)
        
        results = []
        for chunk_result in chunk_results:
            if isinstance(chunk_result, list):
                results.extend(chunk_result)
        
        return results
    
    async def _process_single_batch(self, offset: int, batch_size: int, executor: ThreadPoolExecutor) -> int:
        """Process single batch"""
        students = await self.db.fetch_students_batch(offset, batch_size)
        if not students:
            return 0
        
        results = await self._process_students_parallel(students, executor)
        processed = await self.db.batch_upsert_embeddings(results)
        
        self.stats.total_processed += processed
        self.stats.total_failed += len(students) - processed
        self.stats.batches_completed += 1
        
        return processed
    
    async def import_one_batch(self, offset=0, batch_size=100):
        """Import single batch with legacy logging for backward compatibility"""
        print(f"Starting import for batch at offset {offset} with batch size {batch_size}")
        print("Connecting to PostgreSQL database...")
        
        conn = await self.db.get_connection()
        try:
            print(f"Fetching students from database - Offset {offset}, Batch size {batch_size}")
            
            # Use legacy method for single batch
            query = """
                SELECT student_id, COALESCE(pen, 'NULL') as pen, COALESCE(legal_first_name, 'NULL') as legal_first_name,
                       COALESCE(legal_last_name, 'NULL') as legal_last_name, COALESCE(legal_middle_names, 'NULL') as legal_middle_names,
                       COALESCE(dob::text, 'NULL') as dob, COALESCE(sex_code, 'NULL') as sex_code,
                       COALESCE(postal_code, 'NULL') as postal_code, COALESCE(mincode, 'NULL') as mincode,
                       COALESCE(LPAD(local_id::text, 8, '0'), 'NULL') as local_id
                FROM "api_pen_match_v2".student ORDER BY student_id ASC LIMIT $1 OFFSET $2
            """
            rows = await conn.fetch(query, batch_size, offset)
            
            students = []
            for row in rows:
                student = {
                    "student_id": row["student_id"], "pen": row["pen"], "legalFirstName": row["legal_first_name"],
                    "legalLastName": row["legal_last_name"], "legalMiddleNames": row["legal_middle_names"],
                    "dob": row["dob"], "sexCode": row["sex_code"], "postalCode": row["postal_code"],
                    "mincode": row["mincode"], "localID": row["local_id"]
                }
                students.append(student)
                print(f"Fetched student: {student['student_id']} - {student['pen']} - {student['legalFirstName']} {student['legalLastName']}")
            
            if not students:
                print(f"No students found at offset {offset} - Import completed")
                return 0
            
            print(f"Starting embedding generation for {len(students)} students...")
            processed = 0
            
            for i, student in enumerate(students, 1):
                try:
                    student_id = student.get("student_id")
                    if not student_id:
                        continue
                        
                    print(f"Processing student {i}/{len(students)} - ID: {student_id}")
                    embedding = self.student_embedding.generate_embedding(student)
                    
                    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                    await conn.execute("""
                        INSERT INTO "api_pen_match_v2".student_embeddings (student_id, embedding, status_code, create_user, update_user)
                        VALUES ($1, $2, $3, $4, $5) ON CONFLICT (student_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding, update_user = EXCLUDED.update_user, update_date = now()
                    """, student_id, embedding_str, 'A', 'system', 'system')
                    processed += 1
                    print(f"Successfully processed student {student_id}")
                    
                except Exception as e:
                    print(f"Error processing student {student.get('student_id', 'unknown')}: {e}")
                    continue
            
            print(f"Batch import completed - {processed}/{len(students)} students processed")
            return processed
            
        except Exception as e:
            print(f"Batch import failed: {e}")
            raise
        finally:
            await conn.close()
    
    async def import_all_students(self, batch_size=1000):
        """Optimized import for all students"""
        print(f"Starting optimized import for all students (batch_size: {batch_size})")
        
        self.stats.start_time = time.time()
        await self.db.create_pool()
        
        try:
            total_count = await self.db.get_total_student_count()
            print(f"Total records: {total_count:,}")
            
            with ThreadPoolExecutor(max_workers=self.thread_pool_size) as executor:
                semaphore = asyncio.Semaphore(self.max_concurrent_batches)
                
                async def process_batch_with_semaphore(offset):
                    async with semaphore:
                        return await self._process_single_batch(offset, batch_size, executor)
                
                batch_offsets = list(range(0, total_count, batch_size))
                total_batches = len(batch_offsets)
                print(f"Processing {total_batches:,} batches...")
                
                # Process in chunks of 100 batches
                chunk_size = 100
                for chunk_start in range(0, total_batches, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_batches)
                    chunk_offsets = batch_offsets[chunk_start:chunk_end]
                    
                    tasks = [process_batch_with_semaphore(offset) for offset in chunk_offsets]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Progress update every 50 batches
                    if self.stats.batches_completed % 50 == 0:
                        elapsed = time.time() - self.stats.start_time
                        rate = self.stats.batches_completed / elapsed if elapsed > 0 else 0
                        print(f"Progress: {self.stats.batches_completed:,}/{total_batches:,} batches "
                              f"({self.stats.total_processed:,} records, {rate:.1f} batches/sec)")
            
            elapsed = time.time() - self.stats.start_time
            print(f"Completed: {self.stats.total_processed:,} processed, {self.stats.total_failed:,} failed "
                  f"in {elapsed:.1f}s ({self.stats.total_processed/elapsed:.0f} records/sec)")
            
            return self.stats.total_processed
            
        except Exception as e:
            print(f"Import failed: {e}")
            raise
        finally:
            await self.db.close()

if __name__ == "__main__":
    import sys
    
    async def main():
        print("Starting Embedding Import Service")
        service = EmbeddingImportService()
        
        try:
            if len(sys.argv) > 1:
                if sys.argv[1] == "all":
                    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
                    print(f"Mode: Import all students with batch size {batch_size}")
                    result = await service.import_all_students(batch_size)
                    print(f"Final result: {result} total students processed")
                else:
                    offset = int(sys.argv[1])
                    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
                    print(f"Mode: Import single batch at offset {offset} with batch size {batch_size}")
                    result = await service.import_one_batch(offset, batch_size)
                    print(f"Final result: {result} students processed from offset {offset}")
            else:
                print("Mode: Default import (offset 0, size 100)")
                result = await service.import_one_batch()
                print(f"Final result: {result} students processed from offset 0")
                
            print("Embedding Import Service completed successfully")
            
        except Exception as e:
            print(f"Embedding Import Service failed: {e}")
            return 1
        
        return 0
    
    exit_code = asyncio.run(main())
    exit(exit_code)