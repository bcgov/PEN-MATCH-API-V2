from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from typing import Dict, Any
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.student_match import StudentWorkflow

class PENMatchAgent:
    def __init__(self):
        self.workflow = StudentWorkflow()
        self.llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        self.tools = self._create_tools()
        self.agent = self._create_agent()

    def _create_tools(self):
        """Create tools from existing StudentWorkflow methods"""
        
        def check_name_exists(query: str) -> str:
            """Check if student name exists in Cosmos DB"""
            try:
                data = json.loads(query)
                first_name = data.get("legalFirstName")
                last_name = data.get("legalLastName")
                
                exists = self.workflow.cosmos_client.name_exists(first_name, last_name)
                return f"Name exists: {exists}"
            except Exception as e:
                return f"Error: {str(e)}"

        def process_student_query(query: str) -> str:
            """Process complete student matching workflow"""
            try:
                query_student = json.loads(query)
                result = self.workflow.process_student_query(query_student)
                return json.dumps(result, indent=2)
            except Exception as e:
                return f"Error processing query: {str(e)}"

        def get_candidates(query: str) -> str:
            """Get candidate students by name"""
            try:
                data = json.loads(query)
                first_name = data.get("legalFirstName")
                last_name = data.get("legalLastName")
                
                candidates = self.workflow.cosmos_client.get_students_by_name(first_name, last_name)
                return f"Found {len(candidates)} candidates"
            except Exception as e:
                return f"Error: {str(e)}"

        def find_best_match(query: str) -> str:
            """Find best match using similarity"""
            try:
                data = json.loads(query)
                query_student = data.get("query_student")
                candidates = data.get("candidates")
                
                best_match, score = self.workflow.find_perfect_match(query_student, candidates)
                
                if best_match:
                    return f"Perfect match found! Student ID: {best_match.get('studentID', 'N/A')}, Score: {score:.4f}"
                else:
                    return f"No perfect match. Best score: {score:.4f}"
            except Exception as e:
                return f"Error: {str(e)}"

        return [
            Tool(
                name="CheckNameExists",
                func=check_name_exists,
                description="Check if student name exists. Input: JSON with legalFirstName and legalLastName"
            ),
            Tool(
                name="ProcessStudentQuery", 
                func=process_student_query,
                description="Complete student matching workflow. Input: JSON with student data"
            ),
            Tool(
                name="GetCandidates",
                func=get_candidates,
                description="Get candidate students by name. Input: JSON with legalFirstName and legalLastName"
            ),
            Tool(
                name="FindBestMatch",
                func=find_best_match,
                description="Find best match using similarity. Input: JSON with query_student and candidates"
            )
        ]

    def _create_agent(self):
        """Create the React agent"""
        prompt = PromptTemplate.from_template("""
        You are a PEN (Personal Education Number) matching expert. Help find the correct student record.

        You have access to these tools:
        {tools}

        Use the following format:
        Question: the input question you must answer
        Thought: think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (repeat Thought/Action/Action Input/Observation as needed)
        Thought: I now know the final answer
        Final Answer: the final answer with student ID if found

        Question: {input}
        {agent_scratchpad}
        """)

        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=5)

    def match_student(self, query_student: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to match student"""
        
        query_text = f"""
        Find a match for this student:
        {json.dumps(query_student, indent=2)}
        
        Follow this workflow:
        1. First use ProcessStudentQuery to run the complete workflow
        2. If result shows perfect_match_found, return the student ID
        3. If no perfect match but candidates found, analyze further
        4. Return final result with student ID or no match indication
        """

        try:
            result = self.agent.invoke({"input": query_text})
            return self._parse_result(result, query_student)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "student_id": None
            }

    def _parse_result(self, agent_result: Dict, original_query: Dict) -> Dict[str, Any]:
        """Parse agent result"""
        output = agent_result.get("output", "")
        
        # Look for student ID in output
        import re
        student_id_pattern = r'(?:student[_\s]*id|studentID)[:\s]*([a-zA-Z0-9\-_]+)'
        student_id_match = re.search(student_id_pattern, output, re.IGNORECASE)
        
        # Check for success indicators
        success_indicators = ["perfect match found", "match found", "student id"]
        is_success = any(indicator in output.lower() for indicator in success_indicators)
        
        if is_success and student_id_match:
            return {
                "success": True,
                "student_id": student_id_match.group(1),
                "message": "Match found",
                "details": output
            }
        else:
            return {
                "success": False,
                "student_id": None,
                "message": "No suitable match found",
                "details": output
            }

# Simple usage example
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
    
    print("Testing PEN Match Agent...")
    result = agent.match_student(test_query)
    print(f"Result: {json.dumps(result, indent=2)}")