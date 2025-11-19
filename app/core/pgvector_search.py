import asyncio
from typing import List, Dict, Any
from core.student_embedding import StudentEmbedding
from database.postgresql import PostgreSQLManager

class PGVectorSearchService:
    def __init__(self):
        self.student_embedding = StudentEmbedding()
        self.db = PostgreSQLManager()
    
    async def search_students(self, query: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar students using pgvector similarity"""
        
        # Generate embedding for query
        query_embedding = self.student_embedding.generate_embedding(query)
        
        await self.db.create_pool()
        
        try:
            async with self.db.connection_pool.acquire() as conn:
                embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
                
                # Search using cosine similarity
                search_query = """
                    SELECT 
                        s.student_id,
                        s.pen,
                        s.legal_first_name,
                        s.legal_last_name,
                        s.dob,
                        s.sex_code,
                        s.postal_code,
                        (1 - (se.embedding::vector <=> $1::vector)) as similarity_score
                    FROM "api_pen_match_v2".student_embeddings se
                    JOIN "api_pen_match_v2".student s ON se.student_id = s.student_id
                    WHERE se.status_code = 'A'
                    ORDER BY se.embedding::vector <=> $1::vector
                    LIMIT $2
                """
                
                rows = await conn.fetch(search_query, embedding_str, limit)
                
                return [{
                    "student_id": str(row["student_id"]),
                    "pen": row["pen"],
                    "legalFirstName": row["legal_first_name"],
                    "legalLastName": row["legal_last_name"],
                    "dob": str(row["dob"]) if row["dob"] else None,
                    "sexCode": row["sex_code"],
                    "postalCode": row["postal_code"],
                    "similarity_score": float(row["similarity_score"])
                } for row in rows]
                
        finally:
            await self.db.close()

# Test example
if __name__ == "__main__":
    async def test():
        service = PGVectorSearchService()
        
        # Search query with 2+ fields
        query = {
            "legalFirstName": "EESHVIR",
            "legalLastName": "DENEAULT"
        }
        
        results = await service.search_students(query, limit=3)
        
        print(f"Found {len(results)} matches:")
        for result in results:
            print(f"- {result['legalFirstName']} {result['legalLastName']} "
                  f"(PEN: {result['pen']}, Score: {result['similarity_score']:.3f})")
    
    asyncio.run(test())