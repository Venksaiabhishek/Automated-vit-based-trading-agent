import logging
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import (
    supervisor_node,
    vision_node,
    sentiment_node,
    memory_node,
    strategist_node,
    risk_execution_node
)

logger = logging.getLogger(__name__)

def build_workflow() -> StateGraph:
    """
    Constructs the VIT-Enhanced Agentic Trading System cyclic workflow.
    """
    workflow = StateGraph(AgentState)
    
    # 1. Add all nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("vision", vision_node)
    workflow.add_node("sentiment", sentiment_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("safety_circuit", risk_execution_node)
    
    # 2. Define sequence and edges
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "vision")
    workflow.add_edge("vision", "sentiment")
    workflow.add_edge("sentiment", "memory")
    workflow.add_edge("memory", "strategist")
    
    # Strategist has its internal logic. Afterwards, runs through safety check.
    workflow.add_edge("strategist", "safety_circuit")
    workflow.add_edge("safety_circuit", END)
    
    # 3. Compile graph
    app = workflow.compile()
    logger.info("LangGraph Agent Workflow Compiled successfully.")
    
    return app

# Expose compiled system
agentic_system = build_workflow()
