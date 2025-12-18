from langgraph.graph import StateGraph, START, END
from .state import PenMatchState
from .nodes import llm_analyze_node, fetch_candidates_node, decision_router_node
from .llm_client import LLMClient

def build_graph(llm_client: LLMClient):
    """Build the PEN-MATCH LangGraph workflow with conditional routing"""
    # Initialize the StateGraph
    graph = StateGraph(PenMatchState)
    
    # Add nodes
    graph.add_node("fetch", fetch_candidates_node)
    graph.add_node("router", decision_router_node)
    graph.add_node("analyze", lambda state: llm_analyze_node(state, llm_client))
    
    # Add edges
    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", "router")
    
    # Add conditional edges based on router decision
    def route_decision(state):
        route = state.get("route", "end")
        print(f"DEBUG: Routing decision: {route}")
        return route
    
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "llm_analyze": "analyze",
            "end": END
        }
    )
    
    graph.add_edge("analyze", END)
    
    # Compile and return the graph
    return graph.compile()



