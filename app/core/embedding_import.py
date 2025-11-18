import asyncio
import asyncpg
import ssl
from core.student_embedding import StudentEmbedding
from core.student_api import StudentAPI
from config.settings import settings

class EmbeddingImportService:
    def __init__(self):
        print("Initializing EmbeddingImportService...")
        self.student_embedding = StudentEmbedding()
        self.student_api = StudentAPI()
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
    
    def _fetch_students_page(self, page, page_size):
        """Fetch students from API for a specific page"""
        print(f"Fetching students from API - Page {page}, Size {page_size}")
        try:
            # student_api.get_student_page returns the content directly (list of students)
            students = self.student_api.get_student_page(page=page, size=page_size)
            student_count = len(students) if students else 0
            print(f"API fetch successful - Retrieved {student_count} students from page {page}")
            return students
        except Exception as e:
            print(f"API fetch failed for page {page}: {e}")
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
                # Get student ID (pen is the primary identifier in the API)
                student_id = student.get("pen") or student.get("studentID")
                if not student_id:
                    print(f"Skipping student {i}/{len(students)} - No valid ID found")
                    continue
                    
                print(f"Processing student {i}/{len(students)} - ID: {student_id}")
                
                # Generate embedding
                print(f"Generating embedding for student {student_id}...")
                embedding = self.student_embedding.generate_embedding(student)
                print(f"Embedding generated successfully for student {student_id}")
                
                # Insert to database
                print(f"Inserting embedding to database for student {student_id}...")
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
                await conn.execute(
                    """INSERT INTO "api_pen_match_v2".student_embeddings (student_id, embedding) 
                       VALUES ($1, $2) 
                       ON CONFLICT (student_id) DO UPDATE SET 
                       embedding = EXCLUDED.embedding""",
                    student_id,
                    embedding_str
                )
                processed += 1
                print(f"Successfully inserted embedding for student {student_id}")
                
            except Exception as e:
                print(f"Error processing student {student.get('pen', student.get('studentID', 'unknown'))}: {e}")
                continue
        
        print(f"Embedding processing completed - {processed}/{len(students)} students processed successfully")
        return processed
    
    async def import_one_page(self, page=1, page_size=100):
        """Import one page of students with embeddings"""
        print(f"Starting import for page {page} with page size {page_size}")
        
        conn = await self._get_db_connection()
        
        try:
            # Fetch students from API
            students = self._fetch_students_page(page, page_size)
            
            if not students:
                print(f"No students found on page {page} - Import completed")
                return 0
            
            print(f"Starting database import for {len(students)} students...")
            
            # Process and insert students
            processed = await self._process_and_insert_students(conn, students)
            
            if processed == len(students):
                print(f"Page {page} import completed successfully - {processed}/{len(students)} students processed")
            else:
                print(f"Page {page} import completed with errors - {processed}/{len(students)} students processed successfully")
            
            return processed
            
        except Exception as e:
            print(f"Page {page} import failed: {e}")
            raise
        finally:
            await conn.close()
            print("Database connection closed")
    
    async def import_all_pages(self, page_size=100):
        """Import all pages of students using import_one_page"""
        print(f"Starting import for all pages with page size {page_size}")
        
        page = 1
        total_processed = 0
        
        while True:
            try:
                print(f"Processing page {page}...")
                
                # Use import_one_page for each page
                processed = await self.import_one_page(page, page_size)
                
                if processed == 0:  # No students found, end of pages
                    print(f"No more students found at page {page} - All pages import completed")
                    break
                
                total_processed += processed
                print(f"Page {page} completed - Running total: {total_processed} students processed")
                page += 1
                
            except Exception as e:
                print(f"Error processing page {page}: {e}")
                print(f"Skipping page {page} and continuing with next page...")
                page += 1
                continue
        
        print(f"All pages import completed successfully - Total: {total_processed} students processed")
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
                    # Import all pages
                    page_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
                    print(f"Mode: Import all pages with page size {page_size}")
                    result = await service.import_all_pages(page_size)
                    print(f"Final result: {result} total students processed")
                else:
                    # Import specific page
                    page = int(sys.argv[1])
                    page_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
                    print(f"Mode: Import single page {page} with page size {page_size}")
                    result = await service.import_one_page(page, page_size)
                    print(f"Final result: {result} students processed from page {page}")
            else:
                # Default: import page 1 with 100 students
                print("Mode: Default import (page 1, size 100)")
                result = await service.import_one_page()
                print(f"Final result: {result} students processed from page 1")
                
            print("Embedding Import Service completed successfully")
            
        except Exception as e:
            print(f"Embedding Import Service failed: {e}")
            return 1
        
        return 0
    
    exit_code = asyncio.run(main())
    exit(exit_code)