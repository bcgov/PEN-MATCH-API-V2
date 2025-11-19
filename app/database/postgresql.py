import asyncio
import asyncpg
import ssl
from typing import List, Dict, Any
from config.settings import settings

class PostgreSQLManager:
    def __init__(self, max_connections=20):
        self.max_connections = max_connections
        self.connection_pool = None
        
    async def create_pool(self):
        if self.connection_pool is None:
            ssl_context = ssl.create_default_context()
            self.connection_pool = await asyncpg.create_pool(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_db,
                ssl=ssl_context,
                min_size=5,
                max_size=self.max_connections,
                command_timeout=120,
                server_settings={
                    'application_name': 'embedding_import',
                    'tcp_keepalives_idle': '600',
                    'tcp_keepalives_interval': '30',
                    'tcp_keepalives_count': '3',
                }
            )
    
    async def get_connection(self):
        if not self.connection_pool:
            ssl_context = ssl.create_default_context()
            return await asyncpg.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_db,
                ssl=ssl_context
            )
        return self.connection_pool.acquire()
    
    async def fetch_students_batch(self, offset: int, batch_size: int) -> List[Dict[str, Any]]:
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
                   COALESCE(local_id, 'NULL') as local_id,
                   COALESCE(LPAD(local_id::text, 8, '0'), 'NULL') as local_id_padded
            FROM "api_pen_match_v2".student 
            ORDER BY student_id ASC
            LIMIT $1 OFFSET $2
        """
        
        async with self.connection_pool.acquire() as conn:
            rows = await conn.fetch(query, batch_size, offset)
        
        return [{
            "student_id": row[0],
            "pen": row[1],
            "legalFirstName": row[2],
            "legalLastName": row[3],
            "legalMiddleNames": row[4],
            "dob": row[5],
            "sexCode": row[6],
            "postalCode": row[7],
            "mincode": row[8],
            "localID": row[9]
        } for row in rows]
    
    async def batch_upsert_embeddings(self, results: List[Dict[str, Any]]) -> int:
        successful_results = [r for r in results if r.get('success')]
        if not successful_results:
            return 0
        
        async with self.connection_pool.acquire() as conn:
            async with conn.transaction():
                # Create temp table and copy data
                await conn.execute("""
                    CREATE TEMP TABLE temp_embeddings (
                        student_id UUID, embedding TEXT, status_code VARCHAR(10), 
                        create_user VARCHAR(255), update_user VARCHAR(255)
                    )
                """)
                
                copy_data = ''.join([
                    f"{r['student_id']}\t[{','.join(str(x) for x in r['embedding'])}]\tA\tsystem\tsystem\n"
                    for r in successful_results
                ])
                
                await conn.copy_to_table('temp_embeddings', source=copy_data, format='text', delimiter='\t')
                
                # Upsert from temp table
                await conn.execute("""
                    INSERT INTO "api_pen_match_v2".student_embeddings 
                    (student_id, embedding, status_code, create_user, update_user)
                    SELECT student_id, embedding, status_code, create_user, update_user
                    FROM temp_embeddings
                    ON CONFLICT (student_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding, update_user = EXCLUDED.update_user, update_date = now()
                """)
        
        return len(successful_results)
    
    async def get_total_student_count(self) -> int:
        async with self.connection_pool.acquire() as conn:
            return await conn.fetchval('SELECT COUNT(*) FROM "api_pen_match_v2".student')
    
    async def close(self):
        if self.connection_pool:
            await self.connection_pool.close()