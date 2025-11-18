import asyncio
import asyncpg
import ssl
from core.student_embedding import StudentEmbedding
from config.settings import settings

class EmbeddingImportService:
    def __init__(self):
        print("Initializing EmbeddingImportService...")
        self.student_embedding = StudentEmbedding()
        print("EmbeddingImportService initialized successfully")
        
    async def _get_db_connection(self):
        """Get database connection"""
        print("Connecting to PostgreSQL database...")
        try:
            ssl_context = ssl.create_default_context()
            conn = await asyncpg.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_db,
                ssl=ssl_context
            )
            print("Database connection established successfully")
            return conn
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            raise
    
    async def _fetch_students_batch(self, conn, offset, batch_size):
        """Fetch students from database for a specific batch"""
        print(f"Fetching students from database - Offset {offset}, Batch size {batch_size}")
        try:
            query = """
                SELECT student_id, pen, legal_first_name, legal_last_name, legal_middle_names, 
                       dob, sex_code, postal_code, mincode, local_id
                FROM "api_pen_match_v2".student 
                ORDER BY student_id
                LIMIT $1 OFFSET $2
            """
            rows = await conn.fetch(query, batch_size, offset)
            
            # Convert rows to list of dictionaries
            students = []
            for row in rows:
                student = {
                    "student_id": row["student_id"],  # Keep original UUID
                    "pen": row["pen"],
                    "legalFirstName": row["legal_first_name"],
                    "legalLastName": row["legal_last_name"],
                    "legalMiddleNames": row["legal_middle_names"],
                    "dob": str(row["dob"]) if row["dob"] else None,
                    "sexCode": row["sex_code"],
                    "postalCode": row["postal_code"],
                    "mincode": row["mincode"],
                    "localID": row["local_id"]
                }
                students.append(student)
            
            student_count = len(students)
            print(f"Database fetch successful - Retrieved {student_count} students from offset {offset}")
            return students
        except Exception as e:
            print(f"Database fetch failed for offset {offset}: {e}")
            raise
    
    async def _process_and_insert_students(self, conn, students):
        """Process students and insert embeddings to database"""
        if not students:
            print("No students to process")
            return 0
            
        print(f"Starting embedding generation for {len(students)} students...")
        processed = 0
        
        for i, student in enumerate(students, 1):
            try:
                # Use original student_id from database
                student_id = student.get("student_id")
                pen = student.get("pen")
                
                if not student_id:
                    print(f"Skipping student {i}/{len(students)} - No valid student_id found")
                    continue
                    
                print(f"Processing student {i}/{len(students)} - ID: {student_id}, PEN: {pen}")
                
                # Generate embedding using only specified fields
                print(f"Generating embedding for student {student_id}...")
                embedding = self.student_embedding.generate_embedding(student)
                print(f"Embedding generated successfully for student {student_id}")
                
                # Insert to database using original student_id
                print(f"Inserting embedding to database for student {student_id}...")
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                await conn.execute("""
                    INSERT INTO "api_pen_match_v2".student_embeddings (student_id, embedding, status_code, create_user, update_user)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (student_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    update_user = EXCLUDED.update_user,
                    update_date = now()
                """, student_id, embedding_str, 'A', 'system', 'system')
                processed += 1
                print(f"Successfully inserted embedding for student {student_id}")
                
            except Exception as e:
                print(f"Error processing student {student.get('student_id', 'unknown')}: {e}")
                continue
        
        print(f"Embedding processing completed - {processed}/{len(students)} students processed successfully")
        return processed
    
    async def import_one_batch(self, offset=0, batch_size=100):
        """Import one batch of students with embeddings"""
        print(f"Starting import for batch at offset {offset} with batch size {batch_size}")
        
        conn = await self._get_db_connection()
        
        try:
            # Fetch students from database
            students = await self._fetch_students_batch(conn, offset, batch_size)
            
            if not students:
                print(f"No students found at offset {offset} - Import completed")
                return 0
            
            print(f"Starting database import for {len(students)} students...")
            
            # Process and insert students
            processed = await self._process_and_insert_students(conn, students)
            
            if processed == len(students):
                print(f"Batch at offset {offset} import completed successfully - {processed}/{len(students)} students processed")
            else:
                print(f"Batch at offset {offset} import completed with errors - {processed}/{len(students)} students processed successfully")
            
            return processed
            
        except Exception as e:
            print(f"Batch at offset {offset} import failed: {e}")
            raise
        finally:
            await conn.close()
            print("Database connection closed")
    
    async def import_all_students(self, batch_size=100):
        """Import all students using batches"""
        print(f"Starting import for all students with batch size {batch_size}")
        
        offset = 0
        total_processed = 0
        
        while True:
            try:
                print(f"Processing batch at offset {offset}...")
                
                # Import one batch
                processed = await self.import_one_batch(offset, batch_size)
                
                if processed == 0:  # No students found, end of data
                    print(f"No more students found at offset {offset} - All students import completed")
                    break
                
                total_processed += processed
                print(f"Batch at offset {offset} completed - Running total: {total_processed} students processed")
                offset += batch_size
                
            except Exception as e:
                print(f"Error processing batch at offset {offset}: {e}")
                print(f"Skipping batch at offset {offset} and continuing with next batch...")
                offset += batch_size
                continue
        
        print(f"All students import completed successfully - Total: {total_processed} students processed")
        return total_processed

# Run the import
if __name__ == "__main__":
    import sys
    
    async def main():
        print("Starting Embedding Import Service")
        
        service = EmbeddingImportService()
        
        try:
            if len(sys.argv) > 1:
                if sys.argv[1] == "all":
                    # Import all students
                    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
                    print(f"Mode: Import all students with batch size {batch_size}")
                    result = await service.import_all_students(batch_size)
                    print(f"Final result: {result} total students processed")
                else:
                    # Import specific batch
                    offset = int(sys.argv[1])
                    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
                    print(f"Mode: Import single batch at offset {offset} with batch size {batch_size}")
                    result = await service.import_one_batch(offset, batch_size)
                    print(f"Final result: {result} students processed from offset {offset}")
            else:
                # Default: import first batch with 100 students
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