import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting VIT-Enhanced Agentic Trading System")
    load_dotenv()
    
    # Check for required API keys
    if not os.getenv('GEMINI_API_KEY'):
        logger.warning("GEMINI_API_KEY missing. Core logic requires Gemini to function.")
        logger.warning("Please configure your .env file with GEMINI_API_KEY before running the system.")
        return

    logger.info("API Keys verified. Initializing trading core...")
    
    # Import agent system after env is loaded
    from src.agent.graph import agentic_system
    
    test_ticker = "AAPL"
    
    logger.info(f"Running automated multi-modal analysis for {test_ticker}...")
    try:
        initial_state = {
            "ticker": test_ticker,
            "tickers": [test_ticker],
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
        
        print(f"\n{'='*60}")
        print(f"  FINAL STRATEGY DECISION for {test_ticker}")
        print(f"{'='*60}")
        print(f"  Decision:    {output.get('strategist_decision')}")
        print(f"  Confidence:  {output.get('strategist_confidence')}")
        print(f"  Execution:   {output.get('final_execution_status')}")
        print(f"  Conflict:    {'YES — Deep Search was used' if output.get('signals_conflict') else 'No — Signals aligned'}")
        print(f"{'='*60}")
        
        print(f"\n--- Agent Thought Process ---")
        for i, thought in enumerate(output.get('thought_process', []), 1):
            print(f"  [{i}] {thought}")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
