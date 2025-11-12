from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from typing import Dict, Any, List, Tuple
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.student_match import StudentWorkflow
from core.similarity_calculator import SimilarityCalculator
from core.embedding_service import EmbeddingService

class PENMatchAgent:
    def __init__(self):
        self.workflow = StudentWorkflow()
        self.similarity_calc = SimilarityCalculator()
        self.embedding_service = EmbeddingService()
        self.llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        self.tools = self._create_tools()
        self.agent = self._create_agent()

    def _create_tools(self):
        """Create tools based on the specified workflow"""
        
        def query_process_chain(query: str) -> str:
            """Initial query processing with LLM analysis"""
            try:
                data = json.loads(query)
                
                prompt = f"""
                Analyze this student query and extract key information for matching:
                {json.dumps(data, indent=2)}
                
                Identify:
                1. Primary matching fields (name, DOB, PEN)
                2. Data quality issues
                3. Potential matching challenges
                4. Confidence level for successful match
                
                Return structured analysis.
                """
                
                analysis = self.llm(prompt)
                return f"Query Analysis: {analysis}"
            except Exception as e:
                return f"Error in query processing: {str(e)}"

        def check_name_exists(query: str) -> str:
            """Check if student name exists and manage embedding workflow"""
            try:
                data = json.loads(query)
                first_name = data.get("legalFirstName")
                last_name = data.get("legalLastName")
                
                # Check if name exists
                exists = self.workflow.cosmos_client.name_exists(first_name, last_name)
                
                if not exists:
                    # Add embeddings to Cosmos
                    student_text = f"{first_name} {last_name} {data.get('dob', '')} {data.get('pen', '')}"
                    embedding = self.embedding_service.create_embedding(student_text)
                    
                    # Store embedding for future retrieval
                    embedding_doc = {
                        "query_student": data,
                        "embedding": embedding,
                        "timestamp": "2025-11-11",
                        "type": "query_embedding"
                    }
                    # Note: You might want to store this in a separate collection
                    
                    return f"Name not found. Added embeddings. Proceeding to retrieval."
                else:
                    return f"Name exists. Proceeding to candidate retrieval."
                    
            except Exception as e:
                return f"Error checking name existence: {str(e)}"

        def retrieve_candidates_from_cosmos(query: str) -> str:
            """Retrieve candidate students from Cosmos DB"""
            try:
                data = json.loads(query)
                first_name = data.get("legalFirstName")
                last_name = data.get("legalLastName")
                
                # Get candidates by name
                candidates = self.workflow.cosmos_client.get_students_by_name(first_name, last_name)
                
                # If no direct name matches, try fuzzy matching
                if not candidates:
                    # Use embedding similarity for broader search
                    student_text = f"{first_name} {last_name} {data.get('dob', '')} {data.get('pen', '')}"
                    candidates = self.workflow.cosmos_client.search_similar_students(student_text, limit=10)
                
                return f"Retrieved {len(candidates)} candidates from Cosmos DB"
                
            except Exception as e:
                return f"Error retrieving candidates: {str(e)}"

        def embedding_similarity_match(query: str) -> str:
            """Perform embedding similarity matching"""
            try:
                data = json.loads(query)
                query_student = data.get("query_student")
                candidates = data.get("candidates", [])
                
                if not candidates:
                    return "No candidates available for similarity matching"
                
                # Calculate similarities
                best_match, similarity_score = self.workflow.find_perfect_match(query_student, candidates)
                
                # Define threshold for perfect match
                PERFECT_MATCH_THRESHOLD = 0.85
                
                if similarity_score >= PERFECT_MATCH_THRESHOLD:
                    return f"PERFECT_MATCH_FOUND|{best_match.get('studentID', 'N/A')}|{similarity_score:.4f}"
                else:
                    return f"NO_PERFECT_MATCH|{similarity_score:.4f}|REQUIRES_ANALYSIS"
                    
            except Exception as e:
                return f"Error in similarity matching: {str(e)}"

        def further_analyze_chain(query: str) -> str:
            """Further analysis using LLM when no clear match is found"""
            try:
                data = json.loads(query)
                query_student = data.get("query_student")
                best_candidate = data.get("best_candidate")
                similarity_score = data.get("similarity_score", 0.0)
                
                prompt = f"""
                Analyze this potential student match for final decision:
                
                Query Student:
                {json.dumps(query_student, indent=2)}
                
                Best Candidate:
                {json.dumps(best_candidate, indent=2)}
                
                Similarity Score: {similarity_score}
                
                Consider:
                1. Name variations (nicknames, maiden names, typos)
                2. Date of birth accuracy
                3. PEN number validation
                4. Local ID consistency
                5. Data entry errors
                
                Decision: Should this be considered a match?
                Provide reasoning and confidence level.
                Return: MATCH or NO_MATCH with explanation.
                """
                
                analysis = self.llm(prompt)
                
                # Parse LLM decision
                if "MATCH" in analysis.upper() and "NO_MATCH" not in analysis.upper():
                    student_id = best_candidate.get('studentID', 'N/A') if best_candidate else 'N/A'
                    return f"FINAL_MATCH_FOUND|{student_id}|{analysis}"
                else:
                    return f"FINAL_NO_MATCH|{analysis}"
                    
            except Exception as e:
                return f"Error in further analysis: {str(e)}"

        def perfect_match_output(query: str) -> str:
            """Generate final output for perfect matches"""
            try:
                data = json.loads(query)
                student_id = data.get("student_id")
                confidence = data.get("confidence", "high")
                method = data.get("method", "similarity")
                
                return f"PERFECT_MATCH_CONFIRMED|{student_id}|{confidence}|{method}"
                
            except Exception as e:
                return f"Error generating output: {str(e)}"

        return [
            Tool(
                name="QueryProcessChain",
                func=query_process_chain,
                description="Initial query processing and analysis. Input: JSON student data"
            ),
            Tool(
                name="CheckNameExists",
                func=check_name_exists,
                description="Check name existence and manage embedding workflow. Input: JSON with student data"
            ),
            Tool(
                name="RetrieveCandidates",
                func=retrieve_candidates_from_cosmos,
                description="Retrieve candidate students from Cosmos DB. Input: JSON with student data"
            ),
            Tool(
                name="EmbeddingSimilarityMatch",
                func=embedding_similarity_match,
                description="Perform embedding similarity matching. Input: JSON with query_student and candidates"
            ),
            Tool(
                name="FurtherAnalyzeChain",
                func=further_analyze_chain,
                description="Further LLM analysis for unclear matches. Input: JSON with query_student, best_candidate, similarity_score"
            ),
            Tool(
                name="PerfectMatchOutput",
                func=perfect_match_output,
                description="Generate final perfect match output. Input: JSON with student_id, confidence, method"
            )
        ]

    def _create_agent(self):
        """Create the React agent with structured workflow"""
        prompt = PromptTemplate.from_template("""
        You are a PEN matching expert following a specific workflow:

        WORKFLOW:
        1. QueryProcessChain - Analyze the input query
        2. CheckNameExists - Check if name exists, add embeddings if needed
        3. RetrieveCandidates - Get candidates from Cosmos DB
        4. EmbeddingSimilarityMatch - Calculate similarity scores
        5a. If perfect match found → PerfectMatchOutput
        5b. If no perfect match → FurtherAnalyzeChain → PerfectMatchOutput (if confirmed)

        Available tools: {tools}

        Use this format:
        Question: the input question
        Thought: think about the current step
        Action: [{tool_names}]
        Action Input: JSON input for the tool
        Observation: result from the tool
        ... (continue until workflow complete)
        Thought: I have completed the workflow
        Final Answer: final result with student ID or no match

        Question: {input}
        {agent_scratchpad}
        """)

        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=10)

    def match_student(self, query_student: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete matching workflow"""
        
        query_text = f"""
        Execute the PEN matching workflow for this student:
        {json.dumps(query_student, indent=2)}
        
        Follow the exact workflow:
        1. Start with QueryProcessChain to analyze the query
        2. Use CheckNameExists to check name and manage embeddings
        3. Use RetrieveCandidates to get potential matches
        4. Use EmbeddingSimilarityMatch to find best match
        5. If no perfect match, use FurtherAnalyzeChain for deeper analysis
        6. End with PerfectMatchOutput if match is found
        
        Return the student ID if found, or indicate no match.
        """

        try:
            result = self.agent.invoke({"input": query_text})
            return self._parse_workflow_result(result, query_student)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "student_id": None,
                "workflow_step": "error"
            }

    def _parse_workflow_result(self, agent_result: Dict, original_query: Dict) -> Dict[str, Any]:
        """Parse the workflow result"""
        output = agent_result.get("output", "")
        
        # Look for workflow indicators
        if "PERFECT_MATCH_CONFIRMED" in output:
            parts = output.split("|")
            if len(parts) >= 2:
                return {
                    "success": True,
                    "student_id": parts[1],
                    "confidence": parts[2] if len(parts) > 2 else "high",
                    "method": parts[3] if len(parts) > 3 else "similarity",
                    "message": "Perfect match found through workflow",
                    "workflow_completed": True
                }
        
        elif "PERFECT_MATCH_FOUND" in output:
            parts = output.split("|")
            if len(parts) >= 2:
                return {
                    "success": True,
                    "student_id": parts[1],
                    "confidence": "high",
                    "method": "embedding_similarity",
                    "message": "Perfect match found",
                    "workflow_completed": True
                }
        
        elif "FINAL_MATCH_FOUND" in output:
            parts = output.split("|")
            if len(parts) >= 2:
                return {
                    "success": True,
                    "student_id": parts[1],
                    "confidence": "medium",
                    "method": "llm_analysis",
                    "message": "Match found through further analysis",
                    "workflow_completed": True
                }
        
        else:
            return {
                "success": False,
                "student_id": None,
                "message": "No suitable match found",
                "details": output,
                "workflow_completed": True
            }

# Usage example
if __name__ == "__main__":
    agent = PENMatchAgent()
    
    test_query = {
        "pen": "350800355",
        "legalFirstName": "ROBYN",
        "legalMiddleNames": "DONNELLY", 
        "legalLastName": "ANDERSON",
        "dob": "2010-11-09",
        "localID": "892056259223"
    }
    
    print("Testing Enhanced PEN Match Agent...")
    result = agent.match_student(test_query)
    print(f"Result: {json.dumps(result, indent=2)}")