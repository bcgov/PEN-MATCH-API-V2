import asyncio
from typing import List, Dict, Any
from core.student_embedding import StudentEmbedding
from database.postgresql import PostgreSQLManager

class PGVectorSearchService:
    def __init__(self):
        self.student_embedding = StudentEmbedding()
        self.db = PostgreSQLManager()
    
    async def search_students(self, query: Dict[str, Any], limit: int = 20) -> Dict[str, Any]:
        """
        Search for similar students using pgvector similarity
        
        Args:
            query: Student search fields
            limit: Maximum number of results to return (default 20 for potential candidates)
        
        Returns:
            Dictionary with perfect matches and potential candidates
        """
        
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
                        s.legal_middle_names,
                        s.dob,
                        s.sex_code,
                        s.postal_code,
                        s.mincode,
                        s.local_id,
                        (1 - (se.embedding::vector <=> $1::vector)) as similarity_score
                    FROM "api_pen_match_v2".student_embeddings se
                    JOIN "api_pen_match_v2".student s ON se.student_id = s.student_id
                    WHERE se.status_code = 'A'
                    ORDER BY se.embedding::vector <=> $1::vector
                    LIMIT $2
                """
                
                rows = await conn.fetch(search_query, embedding_str, limit)
                
                all_results = [{
                    "pen": row["pen"],
                    "legalFirstName": row["legal_first_name"],
                    "legalLastName": row["legal_last_name"],
                    "legalMiddleNames": row["legal_middle_names"],
                    "dob": str(row["dob"]) if row["dob"] else None,
                    "sexCode": row["sex_code"],
                    "postalCode": row["postal_code"],
                    "mincode": row["mincode"],
                    "localID": row["local_id"],
                    "similarity_score": float(row["similarity_score"])
                } for row in rows]
                
                # Define thresholds
                PERFECT_MATCH_THRESHOLD = 0.95  # 95% similarity = perfect match
                POTENTIAL_CANDIDATE_THRESHOLD = 0.7  # 70% similarity = potential candidate
                
                # Categorize results
                perfect_matches = [r for r in all_results if r["similarity_score"] >= PERFECT_MATCH_THRESHOLD]
                potential_candidates = [r for r in all_results if POTENTIAL_CANDIDATE_THRESHOLD <= r["similarity_score"] < PERFECT_MATCH_THRESHOLD]
                
                return {
                    "query": query,
                    "thresholds": {
                        "perfect_match": PERFECT_MATCH_THRESHOLD,
                        "potential_candidate": POTENTIAL_CANDIDATE_THRESHOLD
                    },
                    "perfect_matches": perfect_matches,
                    "potential_candidates": potential_candidates,
                    "total_results": len(all_results)
                }
                
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
        
        results = await service.search_students(query, limit=20)
        
        print(f"Query: {results['query']}")
        print(f"Perfect Match Threshold: {results['thresholds']['perfect_match']}")
        print(f"Potential Candidate Threshold: {results['thresholds']['potential_candidate']}")
        print(f"Total Results Found: {results['total_results']}")
        
        print(f"\n=== PERFECT MATCHES ({len(results['perfect_matches'])}) ===")
        for i, match in enumerate(results['perfect_matches'], 1):
            print(f"{i}. {match['legalFirstName']} {match['legalLastName']} {match['legalMiddleNames'] or ''}")
            print(f"   PEN: {match['pen']}, DOB: {match['dob']}, Sex: {match['sexCode']}")
            print(f"   Postal: {match['postalCode']}, Mincode: {match['mincode']}, LocalID: {match['localID']}")
            print(f"   Similarity Score: {match['similarity_score']:.4f}")
            print()
        
        print(f"\n=== POTENTIAL CANDIDATES ({len(results['potential_candidates'])}) ===")
        for i, candidate in enumerate(results['potential_candidates'], 1):
            print(f"{i}. {candidate['legalFirstName']} {candidate['legalLastName']} {candidate['legalMiddleNames'] or ''}")
            print(f"   PEN: {candidate['pen']}, DOB: {candidate['dob']}, Sex: {candidate['sexCode']}")
            print(f"   Postal: {candidate['postalCode']}, Mincode: {candidate['mincode']}, LocalID: {candidate['localID']}")
            print(f"   Similarity Score: {candidate['similarity_score']:.4f}")
            print()
    
    asyncio.run(test())