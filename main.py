import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting VIT-Enhanced Agentic Trading System")
    load_dotenv()
    
    # Check for required API keys (Gemini needed for LLM strategist)
    if not os.getenv('GEMINI_API_KEY'):
        logger.warning("GEMINI_API_KEY missing. Core logic requires Gemini to function.")
        logger.warning("Please configure your .env file with GEMINI_API_KEY before running the system.")
        return

    logger.info("API Keys verified. Initializing trading core...")
    
    # Import agent system after env is loaded
    from src.agent.graph import agentic_system
    
    test_ticker = "AAPL"
    
    logger.info(f"Running automated mock execution for {test_ticker}...")
    try:
        initial_state = {
            "ticker": test_ticker,
            "chart_image_path": "mock_file.png", 
            "vision_signal": "",
            "sentiment_signal": "",
            "memory_signal": "",
            "thought_process": [],
            "strategist_decision": "",
            "strategist_confidence": 0.0,
            "final_execution_status": "",
            "final_quantity": 0.0
        }
        
        output = agentic_system.invoke(initial_state)
        
        print(f"\n[FINAL STRATEGY DECISION for {test_ticker}]")
        print(f"Action: {output.get('strategist_decision')}")
        print(f"Confidence: {output.get('strategist_confidence')}")
        print(f"Execution: {output.get('final_execution_status')}")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
