import operator
from typing import TypedDict, Annotated, List, Any

class AgentState(TypedDict):
    """
    State dictionary matching the LangGraph flow of the VIT-Enhanced Agentic system.
    """
    ticker: str
    chart_image_path: str
    
    # Tool Outputs
    vision_signal: str
    sentiment_signal: str
    memory_signal: str  # Context from the persistent DB
    
    # Internal agent scratchpad / log
    thought_process: Annotated[List[str], operator.add]
    
    # Position Lifecycle Management
    current_position: bool         # Current boolean indicating if we hold the asset
    entry_price: float             # Price we bought it at
    sell_target_date: str          # Date or Datetime string indicating when to evaluate sell
    
    # Strategist Decisions
    strategist_decision: str       # "STRONG BUY", "SELL", "HOLD", "EXTEND"
    strategist_confidence: float   # 0.0 to 1.0
    
    # Final Action (after Risk/Safety Circuit)
    final_execution_status: str    # "APPROVED", "REJECTED"
    final_quantity: float
