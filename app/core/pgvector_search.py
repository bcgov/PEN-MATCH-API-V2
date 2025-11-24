import asyncio
import time
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
            Dictionary with perfect matches and potential candidates
        """
        
        # Time embedding generation
        embedding_start_time = time.time()
        query_embedding = self.student_embedding.generate_embedding(query)
        embedding_time = time.time() - embedding_start_time
        
        await self.db.create_pool()
        
        try:
            async with self.db.connection_pool.acquire() as conn:
                embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
                
                # Define thresholds
                PERFECT_MATCH_THRESHOLD = 0.97
                MIN_THRESHOLD = 0.50  # Lower threshold for potential candidates
                MAX_POTENTIAL_CANDIDATES = 300
                
                # Time perfect match search
                perfect_match_start_time = time.time()
                perfect_match_query = """
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
                        COALESCE(LPAD(s.local_id::text, 8, '0'), 'NULL') as local_id,
                        (1 - (se.embedding::vector <=> $1::vector)) as similarity_score
                    FROM "api_pen_match_v2".student_embeddings se
                    JOIN "api_pen_match_v2".student s ON se.student_id = s.student_id
                    WHERE se.status_code = 'A'
                    AND (1 - (se.embedding::vector <=> $1::vector)) >= $2
                    ORDER BY (1 - (se.embedding::vector <=> $1::vector)) DESC
                """
                
                perfect_matches_rows = await conn.fetch(perfect_match_query, embedding_str, PERFECT_MATCH_THRESHOLD)
                perfect_match_time = time.time() - perfect_match_start_time
                
                perfect_matches = [{
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
                } for row in perfect_matches_rows]
                
                # If no perfect matches, get top 300 potential candidates
                potential_candidates = []
                potential_candidates_time = 0.0
                
                if not perfect_matches:
                    potential_candidates_start_time = time.time()
                    potential_candidates_query = """
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
                            COALESCE(LPAD(s.local_id::text, 8, '0'), 'NULL') as local_id,
                            (1 - (se.embedding::vector <=> $1::vector)) as similarity_score
                        FROM "api_pen_match_v2".student_embeddings se
                        JOIN "api_pen_match_v2".student s ON se.student_id = s.student_id
                        WHERE se.status_code = 'A'
                        AND (1 - (se.embedding::vector <=> $1::vector)) >= $2
                        ORDER BY (1 - (se.embedding::vector <=> $1::vector)) DESC
                        LIMIT $3
                    """
                    
                    potential_rows = await conn.fetch(potential_candidates_query, embedding_str, MIN_THRESHOLD, MAX_POTENTIAL_CANDIDATES)
                    potential_candidates_time = time.time() - potential_candidates_start_time
                    
                    potential_candidates = [{
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
                    } for row in potential_rows]
                
                # Calculate total search time
                total_search_time = perfect_match_time + potential_candidates_time
                
                return {
                    "query": query,
                    "thresholds": {
                        "perfect_match": PERFECT_MATCH_THRESHOLD,
                        "minimum": MIN_THRESHOLD
                    },
                    "perfect_matches": perfect_matches,
                    "potential_candidates": potential_candidates,
                    "total_perfect_matches": len(perfect_matches),
                    "total_potential_candidates": len(potential_candidates),
                    "performance": {
                        "embedding_time_seconds": round(embedding_time, 4),
                        "perfect_match_search_time_seconds": round(perfect_match_time, 4),
                        "potential_candidates_search_time_seconds": round(potential_candidates_time, 4),
                        "total_search_time_seconds": round(total_search_time, 4),
                        "total_processing_time_seconds": round(embedding_time + total_search_time, 4)
                    }
                }
                
        finally:
            await self.db.close()

# Test example
if __name__ == "__main__":
    async def test():
        service = PGVectorSearchService()
        
        # Search query with actual data
        query = {
                "legalFirstName": "",
                "legalLastName": "",
                "legalMiddleNames": "",
                "dob": "",
                "sexCode": "",
                "postalCode": "",
                "mincode": ""
        }
        
        print("Searching for students...")
        results = await service.search_students(query)

        # Performance metrics
        perf = results['performance']
        print(f"\n=== PERFORMANCE METRICS ===")
        print(f"Embedding Generation: {perf['embedding_time_seconds']} seconds")
        print(f"Perfect Match Search: {perf['perfect_match_search_time_seconds']} seconds")
        print(f"Potential Candidates Search: {perf['potential_candidates_search_time_seconds']} seconds")
        print(f"Total Search Time: {perf['total_search_time_seconds']} seconds")
        print(f"Total Processing Time: {perf['total_processing_time_seconds']} seconds")

        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"Perfect Matches (≥97%): {results['total_perfect_matches']}")
        if results['total_perfect_matches'] == 0:
            print(f"Top Potential Candidates: {results['total_potential_candidates']}")
    
        if results['perfect_matches']:
            print(f"\n=== PERFECT MATCHES ({len(results['perfect_matches'])}) ===")
            for i, match in enumerate(results['perfect_matches'], 1):
                print(f"{i}. {match['legalFirstName']} {match['legalLastName']} {match['legalMiddleNames'] or ''}")
                print(f"   PEN: {match['pen']}, DOB: {match['dob']}, Sex: {match['sexCode']}")
                print(f"   Postal: {match['postalCode']}, Mincode: {match['mincode']}, LocalID: {match['localID']}")
                print(f"   Similarity Score: {match['similarity_score']:.4f}")
                print()
        else:
            print(f"\n=== TOP POTENTIAL CANDIDATES ({len(results['potential_candidates'])}) ===")
            for i, candidate in enumerate(results['potential_candidates'][:10], 1):
                print(f"{i}. {candidate['legalFirstName']} {candidate['legalLastName']} {candidate['legalMiddleNames'] or ''}")
                print(f"   PEN: {candidate['pen']}, DOB: {candidate['dob']}, Sex: {candidate['sexCode']}")
                print(f"   Postal: {candidate['postalCode']}, Mincode: {candidate['mincode']}, LocalID: {candidate['localID']}")
                print(f"   Similarity Score: {candidate['similarity_score']:.4f}")
                print()
        

    asyncio.run(test())