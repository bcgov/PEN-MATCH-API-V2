import asyncio
import time
from typing import List, Dict, Any, Optional
from core.student_embedding import StudentEmbedding
from database.postgresql import PostgreSQLManager
import difflib

class PGVectorSearchService:
    def __init__(self):
        self.student_embedding = StudentEmbedding()
        self.db = PostgreSQLManager()
    
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
                    print("Creating HNSW index for improved search performance...")
                    create_index_start = time.time()
                    
                    create_index_query = """
                        CREATE INDEX CONCURRENTLY student_embeddings_hnsw_idx 
                        ON "api_pen_match_v2".student_embeddings 
                        USING hnsw (embedding vector_cosine_ops) 
                        WITH (m = 16, ef_construction = 64)
                    """
                    
                    await conn.execute(create_index_query)
                    create_index_time = time.time() - create_index_start
                    
                    print(f"HNSW index created successfully in {create_index_time:.2f} seconds")
                else:
                    print("HNSW index already exists")
                    
        except Exception as e:
            print(f"Error managing HNSW index: {e}")
        finally:
            await self.db.close()
    
    def _calculate_postal_similarity(self, query_postal: str, candidate_postal: str) -> float:
        """Calculate postal code similarity (unreliable, small weight)"""
        if not query_postal or not candidate_postal:
            return 0.0
        
        # Remove spaces and convert to uppercase
        query_clean = query_postal.replace(" ", "").upper()
        candidate_clean = candidate_postal.replace(" ", "").upper()
        
        if query_clean == candidate_clean:
            return 1.0
        
        # Use sequence matcher for fuzzy matching
        similarity = difflib.SequenceMatcher(None, query_clean, candidate_clean).ratio()
        return similarity
    
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
        return similarity if similarity > 0.7 else 0.0
    
    def _calculate_sex_similarity(self, query_sex: str, candidate_sex: str) -> float:
        """Calculate sex similarity (exact match only)"""
        if not query_sex or not candidate_sex:
            return 0.0
        
        return 1.0 if query_sex.upper() == candidate_sex.upper() else 0.0
    
    def _calculate_soft_score(self, query: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """Calculate soft scoring for postal, mincode, sex"""
        soft_score = 0.0
        
        # Postal code similarity (weight: 0.1 - unreliable)
        postal_sim = self._calculate_postal_similarity(
            query.get("postalCode", ""),
            candidate.get("postalCode", "")
        )
        soft_score += postal_sim * 0.1
        
        # Mincode similarity (weight: 0.15)
        mincode_sim = self._calculate_mincode_similarity(
            query.get("mincode", ""),
            candidate.get("mincode", "")
        )
        soft_score += mincode_sim * 0.15
        
        # Sex similarity (weight: 0.05)
        sex_sim = self._calculate_sex_similarity(
            query.get("sexCode", ""),
            candidate.get("sexCode", "")
        )
        soft_score += sex_sim * 0.05
        
        return soft_score
    
    async def search_students(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hybrid search: Name embedding + soft scoring for other fields
        
        Args:
            query: Student search fields
        
        Returns:
            Dictionary with ranked results
        """
        
        print(f"=== HYBRID SEARCH STARTING ===")
        print(f"Query: {query}")
        
        # Step 1: Generate name embedding
        embedding_start_time = time.time()
        query_embedding = self.student_embedding.generate_embedding(query)
        embedding_time = time.time() - embedding_start_time
        
        query_text = self.student_embedding.student_to_text(query)
        print(f"Name text for embedding: '{query_text}'")
        print(f"Generated embedding with {len(query_embedding)} dimensions")
        
        await self.db.create_pool()
        
        try:
            async with self.db.connection_pool.acquire() as conn:
                # Set HNSW search parameters
                await conn.execute("SET hnsw.ef_search = 200")
                
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
                if query.get("dob") and query["dob"] != 'NULL':
                    print(f"Applying DOB hard filter: {query['dob']}")
                    base_query += " AND se.dob = $2"
                    query_params.append(query["dob"])
                
                # Order by embedding similarity and limit to top 200
                base_query += """
                    ORDER BY se.embedding <=> $1::vector ASC
                    LIMIT 200
                """
                
                print(f"Executing vector search with {'DOB filter' if len(query_params) > 1 else 'no filters'}...")
                candidates_rows = await conn.fetch(base_query, *query_params)
                vector_search_time = time.time() - vector_search_start_time
                
                print(f"Vector search returned {len(candidates_rows)} candidates in {vector_search_time:.4f}s")
                
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
                    
                    scored_candidates.append(candidate)
                
                scoring_time = time.time() - scoring_start_time
                
                # Step 4: Final ranking by combined score
                scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
                
                # Calculate total processing time
                total_time = embedding_time + vector_search_time + scoring_time
                
                print(f"=== HYBRID SEARCH COMPLETED ===")
                print(f"Embedding time: {embedding_time:.4f}s")
                print(f"Vector search time: {vector_search_time:.4f}s")
                print(f"Soft scoring time: {scoring_time:.4f}s")
                print(f"Total time: {total_time:.4f}s")
                
                return {
                    "query": query,
                    "query_text_for_embedding": query_text,
                    "methodology": {
                        "step1": "Name embedding generation",
                        "step2": "SQL hard filter (DOB if provided) + vector search (top 200)",
                        "step3": "Python soft scoring (postal, mincode, sex)",
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
                
        finally:
            await self.db.close()

# Test example
if __name__ == "__main__":
    async def test():
        service = PGVectorSearchService()
        
        # Create HNSW index if needed
        await service.create_hnsw_index_if_not_exists()
        
        # Test with sample query
        query = {
            "legalFirstName": "Michael",
            "legalLastName": "Lee",
            "legalMiddleNames": "James",
            "dob": "1995-08-15",  # Optional DOB hard filter
            "sexCode": "M",
            "postalCode": "V5K2A1",
            "mincode": "12345678"
        }
        
        print("=== TESTING HYBRID SEARCH ===")
        results = await service.search_students(query)

        # Performance metrics
        perf = results['performance']
        print(f"\n=== PERFORMANCE METRICS ===")
        print(f"Query text: '{results['query_text_for_embedding']}'")
        print(f"Embedding Generation: {perf['embedding_time_seconds']} seconds")
        print(f"Vector Search: {perf['vector_search_time_seconds']} seconds")
        print(f"Soft Scoring: {perf['soft_scoring_time_seconds']} seconds")
        print(f"Total Processing: {perf['total_processing_time_seconds']} seconds")

        # Methodology
        print(f"\n=== METHODOLOGY ===")
        for step, desc in results['methodology'].items():
            print(f"{step}: {desc}")

        # Results summary
        print(f"\n=== RESULTS SUMMARY ===")
        print(f"Total Candidates: {results['total_candidates']}")
        print(f"Returning top 10 ranked by: embedding similarity + soft score bonus")

        # Top 10 results
        top_10 = results['candidates'][:10]
        if top_10:
            print(f"\n=== TOP 10 RANKED RESULTS ===")
            for i, candidate in enumerate(top_10, 1):
                print(f"{i}. {candidate['legalFirstName']} {candidate['legalLastName']} {candidate['legalMiddleNames'] or ''}")
                print(f"   PEN: {candidate['pen']}")
                print(f"   DOB: {candidate['dob']}, Sex: {candidate['sexCode']}")
                print(f"   Postal: {candidate['postalCode']}, Mincode: {candidate['mincode']}")
                print(f"   Embedding Similarity: {candidate['embedding_similarity']:.4f}")
                print(f"   Soft Score Bonus: {candidate['soft_score']:.4f}")
                print(f"   Final Score: {candidate['final_score']:.4f}")
                print()
        else:
            print("No candidates found")

        # Output PENs for top 10
        top_10_pens = [candidate['pen'] for candidate in top_10]
        print(f"=== TOP 10 PENS ===")
        print(f"PENs: {top_10_pens}")

    asyncio.run(test())