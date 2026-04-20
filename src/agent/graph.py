"""
LangGraph Agent Workflow — VIT-Enhanced Agentic Trading System
Builds the full conditional graph with:
- Data Fetcher → Supervisor → Vision → Sentiment → Validation
  → [Conflict?] → Deep Search → Memory → Strategist → Safety Circuit → END
  → [Aligned?] → Memory → Strategist → Safety Circuit → END
- Hourly simulation scheduler
"""
import logging
import threading
import time
from typing import Literal

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import (
    data_fetcher_node,
    supervisor_node,
    vision_node,
    sentiment_node,
    validation_node,
    deep_search_node,
    memory_node,
    strategist_node,
    risk_execution_node
)

logger = logging.getLogger(__name__)


def should_deep_search(state: AgentState) -> Literal["deep_search", "memory"]:
    """
    Conditional edge after validation node.
    Routes to deep_search if signals conflict, else directly to memory.
    """
    if state.get('signals_conflict', False):
        logger.info("Routing to Deep Search Agent (conflict detected)")
        return "deep_search"
    else:
        logger.info("Routing to Memory (signals aligned)")
        return "memory"


def build_workflow() -> StateGraph:
    """
    Constructs the VIT-Enhanced Agentic Trading System workflow with conditional edges.
    
    Flow:
    START → data_fetcher → supervisor → vision → sentiment → validation
                                                                ↓
                                                  [conflict?] → deep_search → memory → strategist → safety_circuit → END
                                                  [aligned?]  → memory → strategist → safety_circuit → END
    """
    workflow = StateGraph(AgentState)
    
    # 1. Add all nodes
    workflow.add_node("data_fetcher", data_fetcher_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("vision", vision_node)
    workflow.add_node("sentiment", sentiment_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("deep_search", deep_search_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("safety_circuit", risk_execution_node)
    
    # 2. Define edges
    workflow.add_edge(START, "data_fetcher")
    workflow.add_edge("data_fetcher", "supervisor")
    workflow.add_edge("supervisor", "vision")
    workflow.add_edge("vision", "sentiment")
    workflow.add_edge("sentiment", "validation")
    
    # Conditional: route based on conflict detection
    workflow.add_conditional_edges(
        "validation",
        should_deep_search,
        {
            "deep_search": "deep_search",
            "memory": "memory"
        }
    )
    
    # Deep search flows into memory
    workflow.add_edge("deep_search", "memory")
    
    # Memory → Strategist → Safety → END
    workflow.add_edge("memory", "strategist")
    workflow.add_edge("strategist", "safety_circuit")
    workflow.add_edge("safety_circuit", END)
    
    # 3. Compile graph
    app = workflow.compile()
    logger.info("LangGraph Agent Workflow Compiled successfully (with conditional deep search).")
    
    return app


# Expose compiled system
agentic_system = build_workflow()


# ─── Hourly Simulation Scheduler ───

class HourlySimulator:
    """
    Runs the agent graph on a simulated hourly schedule.
    Each cycle processes all tickers in the watchlist.
    """
    def __init__(self, interval_seconds: int = 3600):
        self.interval = interval_seconds
        self.running = False
        self._timer = None
        self.cycle_count = 0
        self.cycle_logs = []
    
    def start(self, tickers: list[str] = None, simulated: bool = True):
        """Start the hourly simulation loop."""
        if tickers is None:
            tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
        
        self.running = True
        self.tickers = tickers
        
        if simulated:
            # Run immediately for demo, then schedule next
            logger.info(f"Starting hourly simulator with {len(tickers)} tickers (interval: {self.interval}s)")
            self._run_cycle()
        else:
            self._schedule_next()
    
    def stop(self):
        """Stop the simulation loop."""
        self.running = False
        if self._timer:
            self._timer.cancel()
        logger.info("Hourly simulator stopped")
    
    def _schedule_next(self):
        """Schedule the next cycle."""
        if self.running:
            self._timer = threading.Timer(self.interval, self._run_cycle)
            self._timer.daemon = True
            self._timer.start()
    
    def _run_cycle(self):
        """Execute one full cycle across all tickers."""
        self.cycle_count += 1
        logger.info(f"=== Hourly Simulation Cycle #{self.cycle_count} ===")
        
        cycle_results = []
        
        for ticker in self.tickers:
            try:
                initial_state = {
                    "ticker": ticker,
                    "tickers": self.tickers,
                    "chart_image_path": "",
                    "news_headlines": [],
                    "vision_signal": "",
                    "vision_confidence": 0.0,
                    "sentiment_signal": "",
                    "sentiment_score": 0.0,
                    "memory_signal": "",
                    "signals_conflict": False,
                    "deep_search_result": "",
                    "thought_process": [],
                    "current_position": False,
                    "entry_price": 0.0,
                    "sell_target_date": "None",
                    "strategist_decision": "",
                    "strategist_confidence": 0.0,
                    "final_execution_status": "",
                    "final_quantity": 0.0,
                    "session_pnl": 0.0,
                    "initial_capital": 10000.0,
                    "circuit_breaker_triggered": False,
                    "post_mortem_report": ""
                }
                
                output = agentic_system.invoke(initial_state)
                
                cycle_results.append({
                    "ticker": ticker,
                    "decision": output.get('strategist_decision', 'N/A'),
                    "confidence": output.get('strategist_confidence', 0.0),
                    "status": output.get('final_execution_status', 'N/A'),
                    "conflict": output.get('signals_conflict', False),
                    "thoughts": output.get('thought_process', [])
                })
                
                logger.info(f"  {ticker}: {output.get('strategist_decision')} ({output.get('strategist_confidence', 0):.2f})")
                
            except Exception as e:
                logger.error(f"  {ticker}: Cycle failed — {e}")
                cycle_results.append({
                    "ticker": ticker,
                    "decision": "ERROR",
                    "error": str(e)
                })
        
        self.cycle_logs.append({
            "cycle": self.cycle_count,
            "results": cycle_results
        })
        
        # Schedule next cycle
        if self.running:
            self._schedule_next()


# Singleton simulator
hourly_simulator = HourlySimulator(interval_seconds=3600)
