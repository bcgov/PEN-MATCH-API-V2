from typing import Dict, Any
from .pen_graph import build_graph
from .llm_client import LLMClient
from config.settings import settings

class PenMatchWorkflow:
    """Main workflow orchestrator for PEN matching with extensible architecture"""
    
    def __init__(self):
        # Initialize LLM client with clean separation of concerns
        self.llm_client = LLMClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base_o4_mini,
            model="o4-mini"
        )
        
        # Build the workflow graph
        self.graph = build_graph(self.llm_client)
        
        # Future: Add other services here
        # self.validation_service = ValidationService()
        # self.audit_service = AuditService()
        # self.notification_service = NotificationService()
    
    def run_match(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete PEN matching workflow
        
        Args:
            request_data: Student query data (dob, surname, givenName, etc.)
            
        Returns:
            Complete workflow results including decision and confidence
        """
        # Pre-process and validate request
        validated_request = self._validate_request(request_data)
        if not validated_request["valid"]:
            return self._create_error_response(
                request_data, 
                validated_request["error"]
            )
        
        # Initialize workflow state
        initial_state = {"request": request_data}
        
        try:
            # Execute the graph workflow
            result = self.graph.invoke(initial_state)
            
            # Post-process and format response
            return self._format_success_response(request_data, result)
            
        except Exception as e:
            return self._create_error_response(request_data, str(e))
    
    def _validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate incoming request data"""
        # Updated validation for new request format
        required_fields = ["surname", "givenName"]  # At minimum need name fields
        
        for field in required_fields:
            if not request_data.get(field):
                return {
                    "valid": False,
                    "error": f"Missing required field: {field}"
                }
        
        return {"valid": True}
    
    def _format_success_response(self, request_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Format successful workflow response"""
        return {
            "success": True,
            "request": request_data,
            "candidates_count": len(result.get("candidates", [])),
            "search_metadata": result.get("search_metadata"),
            "final_decision": result.get("final_decision"),
            "selected_candidate": result.get("selected_candidate"),
            "confidence": result.get("confidence"),
            "analysis": result.get("analysis"),
            "llm_used": result.get("llm_used", False),
            "model_used": self.llm_client.model,
            "api_base": self.llm_client.base_url or "default",
            "candidates": result.get("candidates", [])[:5]  # Top 5 for review
        }
    
    def _create_error_response(self, request_data: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": error,
            "request": request_data,
            "model_used": self.llm_client.model,
            "api_base": self.llm_client.base_url or "default"
        }

# Factory function for easy import
def create_pen_match_workflow() -> PenMatchWorkflow:
    return PenMatchWorkflow()