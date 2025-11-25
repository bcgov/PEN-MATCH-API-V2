import asyncio
import time
from typing import List, Dict, Any, Optional
from core.student_embedding import StudentEmbedding
from database.postgresql import PostgreSQLManager
import difflib
from datetime import datetime, date

class PGVectorSearchService:
    def __init__(self):
        self.student_embedding = StudentEmbedding()
        self.db = PostgreSQLManager()
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Convert date string to Python date object"""
        if not date_str or date_str == 'NULL':
            return None
        
        try:
            # Parse YYYY-MM-DD format
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return None
    
    async def create_hnsw_index_if_not_exists(self):
        """Create HNSW index on embeddings if it doesn't exist"""
        await self.db.create_pool()
        
        try:
            async with self.db.connection_pool.acquire() as conn:
                check_index_query = """
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE tablename = 'student_embeddings' 
                    AND indexname = 'student_embeddings_hnsw_idx'
                """
                
                existing_index = await conn.fetchval(check_index_query)
                
                if not existing_index:
                    create_index_query = """
                        CREATE INDEX CONCURRENTLY student_embeddings_hnsw_idx 
                        ON "api_pen_match_v2".student_embeddings 
                        USING hnsw (embedding vector_cosine_ops) 
                        WITH (m = 16, ef_construction = 64)
                    """
                    
                    await conn.execute(create_index_query)
                    
        except Exception as e:
            print(f"Error managing HNSW index: {e}")
    
    def _calculate_postal_similarity(self, query_postal: str, candidate_postal: str) -> float:
        """Calculate postal code similarity with improved matching"""
        if not query_postal or not candidate_postal:
            return 0.0
        
        query_clean = query_postal.replace(" ", "").upper()
        candidate_clean = candidate_postal.replace(" ", "").upper()
        
        if query_clean == candidate_clean:
            return 1.0
        
        # Check if first 3 characters match (same area)
        if len(query_clean) >= 3 and len(candidate_clean) >= 3:
            if query_clean[:3] == candidate_clean[:3]:
                return 0.7  # Partial match for same area
        
        # Fuzzy matching for close postal codes
        similarity = difflib.SequenceMatcher(None, query_clean, candidate_clean).ratio()
        return similarity if similarity > 0.5 else 0.0
    
    def _calculate_mincode_similarity(self, query_mincode: str, candidate_mincode: str) -> float:
        """Calculate mincode similarity (exact match preferred)"""
        if not query_mincode or not candidate_mincode:
            return 0.0
        
        query_clean = str(query_mincode).strip()
        candidate_clean = str(candidate_mincode).strip()
        
        if query_clean == candidate_clean:
            return 1.0
        
        # Partial matching for mincode
        similarity = difflib.SequenceMatcher(None, query_clean, candidate_clean).ratio()
        return similarity if similarity > 0.8 else 0.0
    
    def _calculate_sex_similarity(self, query_sex: str, candidate_sex: str) -> float:
        """Calculate sex similarity (exact match only)"""
        if not query_sex or not candidate_sex:
            return 0.0
        
        return 1.0 if query_sex.upper() == candidate_sex.upper() else 0.0
    
    def _has_middle_name_query(self, query: Dict[str, Any]) -> bool:
        """Check if query has middle name"""
        middle = query.get("legalMiddleNames", "")
        return middle and middle.strip() and middle != 'NULL'
    
    def _calculate_soft_score(self, query: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """Calculate soft scoring for postal, mincode, sex with increased weights"""
        soft_score = 0.0
        
        # Check if query has middle name - if not, increase weights for other fields
        has_middle_name = self._has_middle_name_query(query)
        
        if has_middle_name:
            # Normal weights when middle name is provided
            postal_weight = 0.15    # Increased from 0.1
            mincode_weight = 0.20   # Increased from 0.15  
            sex_weight = 0.10       # Increased from 0.05
        else:
            # Higher weights when middle name is missing - compensate for embedding difference
            postal_weight = 0.25    # Much higher weight
            mincode_weight = 0.30   # Much higher weight
            sex_weight = 0.15       # Much higher weight
        
        # Postal code similarity
        postal_sim = self._calculate_postal_similarity(
            query.get("postalCode", ""),
            candidate.get("postalCode", "")
        )
        soft_score += postal_sim * postal_weight
        
        # Mincode similarity  
        mincode_sim = self._calculate_mincode_similarity(
            query.get("mincode", ""),
            candidate.get("mincode", "")
        )
        soft_score += mincode_sim * mincode_weight
        
        # Sex similarity
        sex_sim = self._calculate_sex_similarity(
            query.get("sexCode", ""),
            candidate.get("sexCode", "")
        )
        soft_score += sex_sim * sex_weight
        
        return soft_score
    
    async def search_students(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hybrid search: Name embedding + soft scoring for other fields
        """
        
        # Step 1: Generate name embedding
        embedding_start_time = time.time()
        query_embedding = self.student_embedding.generate_embedding(query)
        embedding_time = time.time() - embedding_start_time
        
        # Ensure database connection pool is available
        if not self.db.connection_pool:
            await self.db.create_pool()
        
        try:
            async with self.db.connection_pool.acquire() as conn:
                # Set HNSW search parameters - increase for better recall when middle names missing
                await conn.execute("SET hnsw.ef_search = 400")  # Increased from 200
                
                embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
                
                # Step 2: Build SQL query with DOB hard filter (if provided)
                vector_search_start_time = time.time()
                
                base_query = """
                    SELECT 
                        s.student_id,
                        s.pen,
                        s.legal_first_name,
                        s.legal_last_name,
                        s.legal_middle_names,
                        se.dob,
                        se.sex_code,
                        se.postal_code,
                        se.mincode,
                        COALESCE(LPAD(s.local_id::text, 8, '0'), 'NULL') as local_id,
                        (1 - (se.embedding <=> $1::vector)) as embedding_similarity,
                        (se.embedding <=> $1::vector) as cosine_distance
                    FROM "api_pen_match_v2".student_embeddings se
                    JOIN "api_pen_match_v2".student s ON se.student_id = s.student_id
                    WHERE se.status_code = 'A'
                """
                
                # Apply DOB hard filter if provided
                query_params = [embedding_str]
                dob_filter_applied = False
                
                if query.get("dob") and query["dob"] != 'NULL':
                    dob_date = self._parse_date(query["dob"])
                    if dob_date:
                        base_query += " AND se.dob = $2"
                        query_params.append(dob_date)
                        dob_filter_applied = True
                
                # Increase limit to get more candidates when middle name is missing
                has_middle_name = self._has_middle_name_query(query)
                limit = 300 if not has_middle_name else 200  # More candidates when middle name missing
                
                # Order by embedding similarity and limit
                base_query += f"""
                    ORDER BY se.embedding <=> $1::vector ASC
                    LIMIT {limit}
                """
                
                candidates_rows = await conn.fetch(base_query, *query_params)
                vector_search_time = time.time() - vector_search_start_time
                
                # Step 3: Python soft scoring
                scoring_start_time = time.time()
                scored_candidates = []
                
                for row in candidates_rows:
                    candidate = {
                        "pen": row["pen"],
                        "legalFirstName": row["legal_first_name"],
                        "legalLastName": row["legal_last_name"],
                        "legalMiddleNames": row["legal_middle_names"],
                        "dob": str(row["dob"]) if row["dob"] else None,
                        "sexCode": row["sex_code"],
                        "postalCode": row["postal_code"],
                        "mincode": row["mincode"],
                        "localID": row["local_id"],
                        "embedding_similarity": float(row["embedding_similarity"]),
                        "cosine_distance": float(row["cosine_distance"])
                    }
                    
                    # Calculate soft score for postal, mincode, sex
                    soft_score = self._calculate_soft_score(query, candidate)
                    
                    # Final score = main signal (embedding) + secondary signal (soft score)
                    final_score = candidate["embedding_similarity"] + soft_score
                    
                    candidate["soft_score"] = soft_score
                    candidate["final_score"] = final_score
                    candidate["has_middle_name_query"] = has_middle_name
                    
                    scored_candidates.append(candidate)
                
                scoring_time = time.time() - scoring_start_time
                
                # Step 4: Final ranking by combined score
                scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
                
                # Calculate total processing time
                total_time = embedding_time + vector_search_time + scoring_time
                
                return {
                    "query": query,
                    "has_middle_name_in_query": has_middle_name,
                    "methodology": {
                        "step1": "Name embedding generation",
                        "step2": f"SQL hard filter (DOB if provided) + vector search (top {limit})",
                        "step3": "Python soft scoring (postal, mincode, sex) - higher weights when no middle name",
                        "step4": "Final ranking (embedding + soft score)"
                    },
                    "candidates": scored_candidates,
                    "total_candidates": len(scored_candidates),
                    "performance": {
                        "embedding_time_seconds": round(embedding_time, 4),
                        "vector_search_time_seconds": round(vector_search_time, 4),
                        "soft_scoring_time_seconds": round(scoring_time, 4),
                        "total_processing_time_seconds": round(total_time, 4)
                    }
                }
        
        except Exception as e:
            print(f"Error during search: {e}")
            raise

# Test example
if __name__ == "__main__":
    async def test():
        service = PGVectorSearchService()
        
        # Create HNSW index if needed
        await service.create_hnsw_index_if_not_exists()
        
        # Test with sample query - NO MIDDLE NAME to test the problem
        query = {
            "legalFirstName": "MICHAEL", 
            "legalLastName": "LEE", 
            # "legalMiddleNames": "",  # No middle name provided
            "sexCode": "M",       
            "postalCode": "V3N1H4",
            "mincode": "05757079"   
        }
        
        print("Searching for students...")
        results = await service.search_students(query)

        # Performance metrics
        perf = results['performance']
        print(f"\n=== PERFORMANCE METRICS ===")
        print(f"Query has middle name: {results['has_middle_name_in_query']}")
        print(f"Embedding Generation: {perf['embedding_time_seconds']} seconds")
        print(f"Vector Search: {perf['vector_search_time_seconds']} seconds")
        print(f"Soft Scoring: {perf['soft_scoring_time_seconds']} seconds")
        print(f"Total Processing: {perf['total_processing_time_seconds']} seconds")

        # Results summary
        print(f"\n=== RESULTS SUMMARY ===")
        print(f"Total Candidates: {results['total_candidates']}")

        # Top 10 results with all fields like pgvector_search_v1.py
        candidates = results['candidates'][:10]
        if candidates:
            print(f"\n=== TOP 10 RANKED RESULTS ===")
            for i, candidate in enumerate(candidates, 1):
                print(f"{i}. {candidate['legalFirstName']} {candidate['legalLastName']} {candidate['legalMiddleNames'] or ''}")
                print(f"   PEN: {candidate['pen']}, DOB: {candidate['dob']}, Sex: {candidate['sexCode']}")
                print(f"   Postal: {candidate['postalCode']}, Mincode: {candidate['mincode']}, LocalID: {candidate['localID']}")
                print(f"   Embedding Similarity: {candidate['embedding_similarity']:.4f}")
                print(f"   Soft Score Bonus: {candidate['soft_score']:.4f}")
                print(f"   Final Score: {candidate['final_score']:.4f}")
                print()
        else:
            print("No candidates found")

        # Close connection pool at the end
        await service.db.close()

    asyncio.run(test())