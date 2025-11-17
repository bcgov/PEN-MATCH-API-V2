import asyncio
import asyncpg
from app.core.student_embedding import StudentEmbedding
from app.api.student_api import StudentAPI
from config.settings import settings

class EmbeddingImportService:
    def __init__(self):
        self.student_embedding = StudentEmbedding()
        self.student_api = StudentAPI()
        
    async def _get_db_connection(self):
        """Get database connection"""
        return await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_db
        )
    
    def _fetch_students_page(self, page, page_size):
        """Fetch students from API for a specific page"""
        return self.student_api.get_students(page=page, page_size=page_size)
    
    async def _process_and_insert_students(self, conn, students):
        """Process students and insert embeddings to database"""
        processed = 0
        for student in students:
            try:
                # Generate embedding
                embedding = self.student_embedding.generate_embedding(student)
                
                # Insert to database (only student_id and embedding)
                student_id = student.get("studentID") or student.get("pen")
                await conn.execute(
                    """INSERT INTO student_embeddings (student_id, embedding) 
                       VALUES ($1, $2) 
                       ON CONFLICT (student_id) DO UPDATE SET 
                       embedding = EXCLUDED.embedding""",
                    student_id,
                    embedding
                )
                processed += 1
                
            except Exception as e:
                print(f"Error processing student {student.get('studentID')}: {e}")
                continue
        
        return processed
    
    async def import_one_page(self, page=1, page_size=100):
        """Import one page of students with embeddings"""
        conn = await self._get_db_connection()
        
        try:
            # Fetch students from API
            data = self._fetch_students_page(page, page_size)
            students = data.get("students", [])
            
            if not students:
                print(f"No students found on page {page}")
                return 0
            
            # Process and insert students
            processed = await self._process_and_insert_students(conn, students)
            
            print(f"Processed page {page} - {processed}/{len(students)} students")
            return processed
            
        finally:
            await conn.close()
    
    async def import_all_pages(self, page_size=100):
        """Import all pages of students using import_one_page"""
        page = 1
        total_processed = 0
        
        while True:
            try:
                # Use import_one_page for each page
                processed = await self.import_one_page(page, page_size)
                
                if processed == 0:  # No students found, end of pages
                    break
                
                total_processed += processed
                page += 1
                
            except Exception as e:
                print(f"Error processing page {page}: {e}")
                page += 1
                continue
        
        print(f"Import completed. Total processed: {total_processed}")
        return total_processed

# Run the import
if __name__ == "__main__":
    import sys
    
    async def main():
        service = EmbeddingImportService()
        
        if len(sys.argv) > 1:
            if sys.argv[1] == "all":
                # Import all pages
                page_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
                await service.import_all_pages(page_size)
            else:
                # Import specific page
                page = int(sys.argv[1])
                page_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
                await service.import_one_page(page, page_size)
        else:
            # Default: import page 1 with 100 students
            await service.import_one_page()
    
    asyncio.run(main())