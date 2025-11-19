import asyncio
from typing import List, Dict, Any
from core.student_embedding import StudentEmbedding
from database.postgresql import PostgreSQLManager

class PGVectorSearchService:
    def __init__(self):
        self.student_embedding = StudentEmbedding()
        self.db = PostgreSQLManager()
    
    async def search_students(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for similar students using pgvector similarity
        
        Args:
            query: Student search fields
        
        Returns:
            Dictionary with perfect matches and potential candidates (all results above threshold)
        """
        
        # Generate embedding for query
        query_embedding = self.student_embedding.generate_embedding(query)
        
        await self.db.create_pool()
        
        try:
            async with self.db.connection_pool.acquire() as conn:
                embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
                
                # Define thresholds
                PERFECT_MATCH_THRESHOLD = 0.95
                POTENTIAL_CANDIDATE_THRESHOLD = 0.80
                MIN_THRESHOLD = 0.70  # Minimum threshold to consider
                
                # Search using cosine similarity - NO LIMIT to get ALL candidates above threshold
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
                    AND (1 - (se.embedding::vector <=> $1::vector)) >= $2
                    ORDER BY (1 - (se.embedding::vector <=> $1::vector)) DESC
                """
                
                rows = await conn.fetch(search_query, embedding_str, MIN_THRESHOLD)
                
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
                
                # Categorize results
                perfect_matches = [r for r in all_results if r["similarity_score"] >= PERFECT_MATCH_THRESHOLD]
                potential_candidates = [r for r in all_results if POTENTIAL_CANDIDATE_THRESHOLD <= r["similarity_score"] < PERFECT_MATCH_THRESHOLD]
                
                return {
                    "query": query,
                    "thresholds": {
                        "perfect_match": PERFECT_MATCH_THRESHOLD,
                        "potential_candidate": POTENTIAL_CANDIDATE_THRESHOLD,
                        "minimum": MIN_THRESHOLD
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
        
        # Search query with actual data
        query = {
                "pen": "",
                "legalFirstName": "",
                "legalLastName": "",
                "legalMiddleNames": "",
                "dob": "",
                "sexCode": "",
                "postalCode": "",
                "mincode": "",
                "localID": ""
        }
        
        print("Searching for ALL candidates above threshold...")
        results = await service.search_students(query)
        
        print(f"Query: {results['query']}")
        print(f"Perfect Match Threshold: {results['thresholds']['perfect_match']}")
        print(f"Potential Candidate Threshold: {results['thresholds']['potential_candidate']}")
        print(f"Minimum Threshold: {results['thresholds']['minimum']}")
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
        
        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"Perfect Matches (≥95%): {len(results['perfect_matches'])}")
        print(f"Potential Candidates (80-94%): {len(results['potential_candidates'])}")
        print(f"Total Candidates (≥70%): {results['total_results']}")
    
    asyncio.run(test())