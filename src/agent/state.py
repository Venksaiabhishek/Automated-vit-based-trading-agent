import operator
from typing import TypedDict, Annotated, List, Any

class AgentState(TypedDict):
    """
    State dictionary matching the LangGraph flow of the VIT-Enhanced Agentic system.
    Extended with fields for deep search, circuit breaker, and backtest support.
    """
    # Core identifiers
    ticker: str
    tickers: List[str]               # List of tickers for multi-ticker scheduling
    chart_image_path: str
    
    # Raw data
    news_headlines: List[str]         # Fetched news headlines
    
    # Tool Outputs
    vision_signal: str
    vision_confidence: float          # ViT confidence score (0.0-1.0)
    sentiment_signal: str
    sentiment_score: float            # FinBERT composite score
    memory_signal: str                # Context from the persistent DB
    
    # Conflict Resolution
    signals_conflict: bool            # Whether ViT and FinBERT disagree
    deep_search_result: str           # Tiebreaker analysis from Deep Search Agent
    
    # Internal agent scratchpad / log
    thought_process: Annotated[List[str], operator.add]
    
    # Position Lifecycle Management
    current_position: bool            # Current boolean indicating if we hold the asset
    entry_price: float                # Price we bought it at
    sell_target_date: str             # Date or Datetime string indicating when to evaluate sell
    
    # Strategist Decisions
    strategist_decision: str          # "STRONG BUY", "SELL", "HOLD", "EXTEND"
    strategist_confidence: float      # 0.0 to 1.0
    
    # Final Action (after Risk/Safety Circuit)
    final_execution_status: str       # "APPROVED", "REJECTED"
    final_quantity: float
    
    # Session Risk Management
    session_pnl: float                # Running P&L for circuit breaker
    initial_capital: float            # Starting capital for drawdown calc
    circuit_breaker_triggered: bool   # Whether 2% drawdown limit was hit
    post_mortem_report: str           # Generated post-mortem if circuit breaker trips
