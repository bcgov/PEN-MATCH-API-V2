import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, date
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
    
    def _parse_date(self, date_str: str) -> date:
        """Convert date string to Python date object"""
        if not date_str or date_str == 'NULL':
            return None
        
        try:
            # Parse YYYY-MM-DD format
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            print(f"Invalid date format: {date_str}")
            return None
    
    def _generate_embeddings_batch(self, students: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for a batch of students with separate columns"""
        results = []
        for student in students:
            try:
                # Generate name-only embedding
                embedding = self.student_embedding.generate_embedding(student)
                
                # Parse date properly
                dob_date = None
                if student.get('dob') and student.get('dob') != 'NULL':
                    dob_date = self._parse_date(student.get('dob'))
                
                # Prepare separate columns
                result = {
                    'student_id': student['student_id'],
                    'embedding': embedding,
                    'dob': dob_date,
                    'postal_code': student.get('postalCode') if student.get('postalCode') != 'NULL' else None,
                    'mincode': student.get('mincode') if student.get('mincode') != 'NULL' else None,
                    'sex_code': student.get('sexCode') if student.get('sexCode') != 'NULL' else None,
                    'success': True
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error generating embedding for student {student.get('student_id')}: {e}")
                results.append({
                    'student_id': student['student_id'], 
                    'embedding': None, 
                    'dob': None,
                    'postal_code': None, 
                    'mincode': None,
                    'sex_code': None,
                    'success': False
                })
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
    
    async def _batch_upsert_embeddings_with_columns(self, results: List[Dict[str, Any]]) -> int:
        """Batch upsert embeddings with separate columns"""
        if not results:
            return 0
        
        successful_results = [r for r in results if r['success'] and r['embedding']]
        if not successful_results:
            return 0
        
        conn = await self.db.get_connection()
        try:
            # Prepare batch insert/update query with all 5 columns
            query = """
                INSERT INTO "api_pen_match_v2".student_embeddings 
                (student_id, embedding, dob, postal_code, mincode, sex_code, status_code, create_user, update_user)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) 
                ON CONFLICT (student_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                dob = EXCLUDED.dob,
                postal_code = EXCLUDED.postal_code,
                mincode = EXCLUDED.mincode,
                sex_code = EXCLUDED.sex_code,
                update_user = EXCLUDED.update_user, 
                update_date = now()
            """
            
            batch_data = []
            for result in successful_results:
                embedding_str = "[" + ",".join(str(x) for x in result['embedding']) + "]"
                batch_data.append((
                    result['student_id'],
                    embedding_str,
                    result['dob'],  # Already converted to date object
                    result['postal_code'], 
                    result['mincode'],
                    result['sex_code'],
                    'A',  # status_code
                    'system',  # create_user
                    'system'   # update_user
                ))
            
            await conn.executemany(query, batch_data)
            return len(successful_results)
            
        except Exception as e:
            print(f"Error in batch upsert: {e}")
            return 0
        finally:
            await conn.close()
    
    async def _process_single_batch(self, offset: int, batch_size: int, executor: ThreadPoolExecutor) -> int:
        """Process single batch with 5-column storage"""
        students = await self.db.fetch_students_batch(offset, batch_size)
        if not students:
            return 0
        
        results = await self._process_students_parallel(students, executor)
        processed = await self._batch_upsert_embeddings_with_columns(results)
        
        self.stats.total_processed += processed
        self.stats.total_failed += len(students) - processed
        self.stats.batches_completed += 1
        
        return processed
    
    async def import_one_batch(self, offset=0, batch_size=100):
        """Import single batch with 5-column storage"""
        print(f"Starting import for batch at offset {offset} with batch size {batch_size}")
        print("Connecting to PostgreSQL database...")
        
        conn = await self.db.get_connection()
        try:
            print(f"Fetching students from database - Offset {offset}, Batch size {batch_size}")
            
            query = """
                SELECT student_id, COALESCE(pen, 'NULL') as pen, 
                       COALESCE(legal_first_name, 'NULL') as legal_first_name,
                       COALESCE(legal_last_name, 'NULL') as legal_last_name, 
                       COALESCE(legal_middle_names, 'NULL') as legal_middle_names,
                       COALESCE(dob::text, 'NULL') as dob, 
                       COALESCE(sex_code, 'NULL') as sex_code,
                       COALESCE(postal_code, 'NULL') as postal_code, 
                       COALESCE(mincode, 'NULL') as mincode,
                       COALESCE(LPAD(local_id::text, 8, '0'), 'NULL') as local_id
                FROM "api_pen_match_v2".student 
                ORDER BY student_id ASC LIMIT $1 OFFSET $2
            """
            rows = await conn.fetch(query, batch_size, offset)
            
            students = []
            for row in rows:
                student = {
                    "student_id": row["student_id"], 
                    "pen": row["pen"], 
                    "legalFirstName": row["legal_first_name"],
                    "legalLastName": row["legal_last_name"], 
                    "legalMiddleNames": row["legal_middle_names"],
                    "dob": row["dob"], 
                    "sexCode": row["sex_code"], 
                    "postalCode": row["postal_code"],
                    "mincode": row["mincode"], 
                    "localID": row["local_id"]
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
                    
                    # Generate name-only embedding
                    embedding = self.student_embedding.generate_embedding(student)
                    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                    
                    # Prepare separate column values with proper date conversion
                    dob_val = self._parse_date(student.get('dob')) if student.get('dob') != 'NULL' else None
                    postal_code_val = student.get('postalCode') if student.get('postalCode') != 'NULL' else None
                    mincode_val = student.get('mincode') if student.get('mincode') != 'NULL' else None
                    sex_code_val = student.get('sexCode') if student.get('sexCode') != 'NULL' else None
                    
                    # Insert with all 5 columns: embedding + 4 separate fields
                    await conn.execute("""
                        INSERT INTO "api_pen_match_v2".student_embeddings 
                        (student_id, embedding, dob, postal_code, mincode, sex_code, status_code, create_user, update_user)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) 
                        ON CONFLICT (student_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        dob = EXCLUDED.dob,
                        postal_code = EXCLUDED.postal_code,
                        mincode = EXCLUDED.mincode,
                        sex_code = EXCLUDED.sex_code,
                        update_user = EXCLUDED.update_user, 
                        update_date = now()
                    """, student_id, embedding_str, dob_val, postal_code_val, mincode_val, sex_code_val, 'A', 'system', 'system')
                    
                    processed += 1
                    print(f"Successfully processed student {student_id} with 5 columns:")
                    print(f"  - Name embedding: {len(embedding)} dimensions")
                    print(f"  - DOB: {dob_val}")
                    print(f"  - Postal Code: {postal_code_val}")
                    print(f"  - Mincode: {mincode_val}")
                    print(f"  - Sex Code: {sex_code_val}")
                    
                except Exception as e:
                    print(f"Error processing student {student.get('student_id', 'unknown')}: {e}")
                    continue
            
            print(f"Batch import completed - {processed}/{len(students)} students processed with 5 columns")
            return processed
            
        except Exception as e:
            print(f"Batch import failed: {e}")
            raise
        finally:
            await conn.close()
    
    async def import_all_students(self, batch_size=1000):
        """Optimized import for all students with 5-column storage"""
        print(f"Starting optimized import for all students with 5-column storage (batch_size: {batch_size})")
        
        self.stats.start_time = time.time()
        await self.db.create_pool()
        
        try:
            total_count = await self.db.get_total_student_count()
            print(f"Total records: {total_count:,}")
            print("Importing 5 columns:")
            print("  1. Name Embedding (First + Last + Middle names)")
            print("  2. DOB")
            print("  3. Postal Code")
            print("  4. Mincode")
            print("  5. Sex Code")
            
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
            print(f"5-Column import completed:")
            print(f"  - Processed: {self.stats.total_processed:,} students")
            print(f"  - Failed: {self.stats.total_failed:,} students")
            print(f"  - Time: {elapsed:.1f}s ({self.stats.total_processed/elapsed:.0f} records/sec)")
            print(f"  - Columns stored: Name Embedding + DOB + Postal Code + Mincode + Sex Code")
            
            return self.stats.total_processed
            
        except Exception as e:
            print(f"5-column import failed: {e}")
            raise
        finally:
            await self.db.close()

    async def import_all_records_by_names(self, target_names: List[tuple]) -> int:
        """Import all student records that match specified name pairs with 5-column storage"""
        print(f"Starting 5-column import for all records matching {len(target_names)} name pairs")
        
        target_names_lower = [(first.lower(), last.lower()) for first, last in target_names]
        
        conn = await self.db.get_connection()
        try:
            total_processed = 0
            total_skipped = 0
            
            for i, (first_name, last_name) in enumerate(target_names, 1):
                print(f"\nProcessing name pair {i}/{len(target_names)}: {first_name} {last_name}")
                
                query = """
                    SELECT student_id, COALESCE(pen, 'NULL') as pen, 
                           COALESCE(legal_first_name, 'NULL') as legal_first_name,
                           COALESCE(legal_last_name, 'NULL') as legal_last_name, 
                           COALESCE(legal_middle_names, 'NULL') as legal_middle_names,
                           COALESCE(dob::text, 'NULL') as dob, 
                           COALESCE(sex_code, 'NULL') as sex_code,
                           COALESCE(postal_code, 'NULL') as postal_code, 
                           COALESCE(mincode, 'NULL') as mincode,
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
                
                processed_for_name = 0
                skipped_for_name = 0
                
                for j, row in enumerate(rows, 1):
                    try:
                        student_id = row["student_id"]
                        print(f"  Processing record {j}/{len(rows)} - Student ID: {student_id}")
                        
                        # Check if embedding already exists
                        existing_check = await conn.fetchval("""
                            SELECT COUNT(*) FROM "api_pen_match_v2".student_embeddings 
                            WHERE student_id = $1
                        """, student_id)
                        
                        if existing_check > 0:
                            print(f"    Embedding already exists for student {student_id} - skipping")
                            skipped_for_name += 1
                            continue
                        
                        # Create student object
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
                            "localID": row["local_id"]
                        }
                        
                        # Generate name-only embedding
                        embedding = self.student_embedding.generate_embedding(student)
                        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                        
                        # Prepare separate column values with proper date conversion
                        dob_val = self._parse_date(student.get('dob')) if student.get('dob') != 'NULL' else None
                        postal_code_val = student.get('postalCode') if student.get('postalCode') != 'NULL' else None
                        mincode_val = student.get('mincode') if student.get('mincode') != 'NULL' else None
                        sex_code_val = student.get('sexCode') if student.get('sexCode') != 'NULL' else None
                        
                        # Insert with all 5 columns
                        await conn.execute("""
                            INSERT INTO "api_pen_match_v2".student_embeddings 
                            (student_id, embedding, dob, postal_code, mincode, sex_code, status_code, create_user, update_user)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                            ON CONFLICT (student_id) DO NOTHING
                        """, student_id, embedding_str, dob_val, postal_code_val, mincode_val, sex_code_val, 'A', 'system', 'system')
                        
                        processed_for_name += 1
                        print(f"    Successfully processed student {student_id} with 5 columns")
                        
                    except Exception as e:
                        print(f"    Error processing student {row.get('student_id', 'unknown')}: {e}")
                        continue
                
                total_processed += processed_for_name
                total_skipped += skipped_for_name
                print(f"Completed {first_name} {last_name}: {processed_for_name} processed, {skipped_for_name} skipped")
            
            print(f"\n5-Column name-based import completed:")
            print(f"  Total processed: {total_processed}")
            print(f"  Total skipped: {total_skipped}")
            print(f"  Columns: Name Embedding + DOB + Postal Code + Mincode + Sex Code")
            
            return total_processed
            
        except Exception as e:
            print(f"5-column name-based import failed: {e}")
            raise
        finally:
            await conn.close()


if __name__ == "__main__":
    import sys
    
    async def main():
        print("Starting Embedding Import Service (5-Column Storage)")
        print("Columns: Name Embedding + DOB + Postal Code + Mincode + Sex Code")
        service = EmbeddingImportService()
        
        try:
            if len(sys.argv) > 1:
                if sys.argv[1] == "all":
                    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
                    print(f"Mode: Import all students with 5-column storage (batch size {batch_size})")
                    result = await service.import_all_students(batch_size)
                    print(f"Final result: {result} total students processed with 5 columns")
                elif sys.argv[1] == "names":
                    target_names = [
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
                    print(f"Mode: Import all records matching {len(target_names)} name pairs with 5-column storage")
                    result = await service.import_all_records_by_names(target_names)
                    print(f"Final result: {result} student records processed with 5 columns")
                else:
                    offset = int(sys.argv[1])
                    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
                    print(f"Mode: Import single batch with 5-column storage (offset {offset}, batch size {batch_size})")
                    result = await service.import_one_batch(offset, batch_size)
                    print(f"Final result: {result} students processed with 5 columns")
            else:
                print("Mode: Default import with 5-column storage (offset 0, size 100)")
                result = await service.import_one_batch()
                print(f"Final result: {result} students processed with 5 columns")
                
            print("5-Column Embedding Import Service completed successfully")
            
        except Exception as e:
            print(f"5-Column Embedding Import Service failed: {e}")
            return 1
        
        return 0
    
    exit_code = asyncio.run(main())
    exit(exit_code)