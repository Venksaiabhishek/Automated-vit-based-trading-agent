import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any

from .state import AgentState
from ..tools.vision import analyze_chart_vision
from ..tools.sentiment import analyze_market_sentiment
from ..memory.vector_store import market_memory
from ..trading.risk import risk_guardrail

logger = logging.getLogger(__name__)

def supervisor_node(state: AgentState):
    """Initializes the multi-modal analysis."""
    ticker = state['ticker']
    logger.info(f"--- Supervisor: Starting Multimodal Analysis for {ticker} ---")
    
    return {
        "thought_process": [f"System: Initiating workflow for {ticker}."]
    }

def vision_node(state: AgentState):
    """Calls the Vision Transformer logic to interpret chart imagery."""
    logger.info("--- Agent 1: Vision Analyst ---")
    
    chart_path = state.get('chart_image_path', '')
    if chart_path:
        vision_result = analyze_chart_vision(chart_path)
    else:
        vision_result = "Vision Agent Error: No chart path provided to analyze."
        
    return {
        "vision_signal": vision_result, 
        "thought_process": [f"Vision Module: {vision_result}"]
    }

def sentiment_node(state: AgentState):
    """Calls the text scraper and extracts semantic layout."""
    logger.info("--- Agent 2: Semantic Sentiment Analyst ---")
    
    ticker = state['ticker']
    sentiment_result = analyze_market_sentiment(ticker)
    
    return {
        "sentiment_signal": sentiment_result, 
        "thought_process": [f"Sentiment Module: Extracted {len(sentiment_result.split())} words of news context."]
    }

def memory_node(state: AgentState):
    """Retrieves long-term market regimes from ChromaDB."""
    logger.info("--- Agent 3: Memory Analyst ---")
    
    current_context = f"{state['vision_signal']} | {state['sentiment_signal']}"
    past_regimes = market_memory.recall_similar_regimes(current_context, n_results=1)
    
    # Use first result or placeholder
    memory_insight = past_regimes[0] if past_regimes else "No relevant past memory."
    
    return {
        "memory_signal": memory_insight,
        "thought_process": [f"Memory Module (ChromaDB): {memory_insight[:100]}..."]
    }

def strategist_node(state: AgentState):
    """
    Lead Strategist Agent powered by Google Gemini API.
    Reconciles vision, sentiment, and memory to construct a final thesis.
    """
    logger.info("--- Agent 4: Lead Strategist (Gemini) ---")
    ticker = state['ticker']
    current_position = state.get('current_position', False)
    sell_target_date = state.get('sell_target_date', 'None')
    
    position_context = f"You CURRENTLY HOLD a position in {ticker}. Target Sell Date: {sell_target_date}." if current_position else f"You DO NOT hold any position in {ticker} currently."
    
    prompt = f"""
    You are the Lead Strategist Agent for a Hedge Fund.
    You must evaluate the following inputs for the ticker {ticker}.
    
    {position_context}
    
    1. Vision Input (Chart Patterns): {state['vision_signal']}
    2. Semantic Sentiment (News Activity): {state['sentiment_signal']}
    3. Memory Input (Historical Context): {state['memory_signal']}
    
    If you DO NOT hold a position:
    Decide whether to BUY or HOLD. If BUY, specify the number of days to hold before re-evaluating risk to sell.
    
    If you CURRENTLY HOLD a position:
    Decide whether to SELL now or EXTEND the hold mathematically. If EXTEND, specify the number of days to extend.
    
    Output exactly three lines:
    Line 1: DECISION: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL / EXTEND]
    Line 2: CONFIDENCE: [A number from 0.0 to 1.0]
    Line 3: TARGET_HOLD_DAYS: [integer]
    """
    
    # Invoke Gemini API
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        response = llm.invoke([
            SystemMessage(content="You are a financial quantitative strategist algorithm."),
            HumanMessage(content=prompt)
        ])
        content = response.content.strip().split('\n')
        
        decision_line = content[0] if len(content) > 0 else "DECISION: HOLD"
        confidence_line = content[1] if len(content) > 1 else "CONFIDENCE: 0.0"
        hold_days_line = content[2] if len(content) > 2 else "TARGET_HOLD_DAYS: 0"
        
        decision = decision_line.split("DECISION:")[-1].strip()
        
        try:
            confidence = float(confidence_line.split("CONFIDENCE:")[-1].strip())
        except ValueError:
            confidence = 0.5
            
        try:
            hold_days = int(hold_days_line.split("TARGET_HOLD_DAYS:")[-1].strip())
        except ValueError:
            hold_days = 0
            
    except Exception as e:
        logger.error(f"Strategist failed to invoke LLM: {e}")
        decision = "HOLD"
        confidence = 0.0
        hold_days = 0

    return {
        "strategist_decision": decision,
        "strategist_confidence": confidence,
        "thought_process": [f"Strategist Module (Gemini): Defined decision '{decision}' with {confidence} confidence. Target Hold: {hold_days} days."]
    }

def risk_execution_node(state: AgentState):
    """
    Deterministic Safety Circuit using Kelly Criterion.
    Receives Strategy's request and validates mathematically.
    """
    logger.info("--- Safety Circuit: Evaluating Kelly Criterion ---")
    decision = state['strategist_decision']
    confidence = state['strategist_confidence']
    
    # Evaluate for Longs/Extends
    if "BUY" not in decision.upper() and "EXTEND" not in decision.upper() and "SELL" not in decision.upper():
        return {
            "final_execution_status": "REJECTED_BY_STRATEGY",
            "final_quantity": 0.0,
            "thought_process": [f"Risk Guardrail: Ignored (Decision was {decision}, no execution required)."]
        }
        
    # Assume we have 10,000 mock capital and the agent wishes to allocate 1,000 based on standard limits
    mock_capital = 10000.0
    requested_amount = 1000.0
    
    if "SELL" in decision.upper():
        # Risk validation could perform profit-taking logic or loss aversion check, but execution allows SELL if requested by Strat
        approved = True
        status = "APPROVED_SELL"
        amount = requested_amount
        reason = "Lead Strategist requested SELL based on timeline/risk."
    else:
        # standard BUY or EXTEND buy risk limits
        validation = risk_guardrail.validate_trade(
            confidence_score=confidence, 
            current_capital=mock_capital, 
            requested_amount=requested_amount
        )
        approved = validation['approved']
        status = "APPROVED_BUY_OR_EXTEND" if approved else "REJECTED_RISK_LIMIT"
        amount = requested_amount if approved else 0.0
        reason = validation['reason']
    
    return {
        "final_execution_status": status,
        "final_quantity": amount,
        "thought_process": [f"Risk Guardrail (Kelly): {reason} => Final Status: {status}"]
    }
