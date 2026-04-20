"""
Agent Nodes — All LangGraph node functions for the trading agent pipeline.
Nodes: data_fetcher → supervisor → vision → sentiment → validation → [deep_search] → memory → strategist → safety_circuit
"""
import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any

from .state import AgentState
from ..tools.vision import analyze_chart_vision, get_vision_signal_parsed
from ..tools.sentiment import analyze_market_sentiment, get_sentiment_parsed
from ..memory.vector_store import market_memory
from ..trading.risk import risk_guardrail, circuit_breaker
from .deep_search import detect_conflict, deep_search_resolve

logger = logging.getLogger(__name__)


# ─── Node 1: Data Fetcher ───

def data_fetcher_node(state: AgentState):
    """
    Automatically pulls the latest news headlines and generates
    a 1-hour/1-day candlestick chart for the ticker.
    """
    ticker = state['ticker']
    logger.info(f"--- Data Fetcher: Pulling data for {ticker} ---")
    
    try:
        from ..data.ingestion import fetch_live_data, generate_candlestick_chart, fetch_news_headlines
        
        # Fetch OHLCV and generate chart
        df = fetch_live_data(ticker, period="1mo", interval="1d")
        chart_path = ""
        if not df.empty:
            chart_path = generate_candlestick_chart(
                ticker, df,
                output_path=f"{ticker}_live_chart.png"
            )
        
        # Fetch news headlines
        articles = fetch_news_headlines(ticker, limit=100)
        headlines = [a.get('headline', '') for a in articles if a.get('headline')]
        
        return {
            "chart_image_path": chart_path if chart_path else state.get('chart_image_path', ''),
            "news_headlines": headlines,
            "thought_process": [
                f"Data Fetcher: Loaded {len(df)} price candles and {len(headlines)} news headlines for {ticker}."
            ]
        }
        
    except Exception as e:
        logger.error(f"Data fetcher failed: {e}")
        return {
            "news_headlines": [],
            "thought_process": [f"Data Fetcher Error: {str(e)}"]
        }


# ─── Node 2: Supervisor ───

def supervisor_node(state: AgentState):
    """Initializes the multi-modal analysis pipeline."""
    ticker = state['ticker']
    logger.info(f"--- Supervisor: Starting Multimodal Analysis for {ticker} ---")
    
    # Check circuit breaker before proceeding
    if state.get('circuit_breaker_triggered', False):
        return {
            "thought_process": [f"Supervisor: ⚠️ Circuit Breaker is ACTIVE for {ticker}. Analysis will proceed but trading is HALTED."]
        }
    
    return {
        "thought_process": [f"Supervisor: Initiating ViT + FinBERT multimodal analysis for {ticker}."]
    }


# ─── Node 3a: Vision Analyst ───

def vision_node(state: AgentState):
    """Calls the Vision Transformer to interpret chart imagery."""
    logger.info("--- Agent 1: Vision Analyst (ViT) ---")
    
    chart_path = state.get('chart_image_path', '')
    if chart_path and os.path.exists(chart_path):
        vision_result = analyze_chart_vision(chart_path)
    else:
        vision_result = "Vision Agent: Unable to analyze — no valid chart image available."
    
    # Parse confidence
    signal, confidence = get_vision_signal_parsed(vision_result)
    
    return {
        "vision_signal": vision_result,
        "vision_confidence": confidence,
        "thought_process": [f"Vision Module (ViT): {vision_result}"]
    }


# ─── Node 3b: Sentiment Analyst ───

def sentiment_node(state: AgentState):
    """Calls FinBERT to analyze market sentiment from news."""
    logger.info("--- Agent 2: Sentiment Analyst (FinBERT) ---")
    
    ticker = state['ticker']
    sentiment_result = analyze_market_sentiment(ticker)
    
    # Parse score
    signal, score = get_sentiment_parsed(sentiment_result)
    
    return {
        "sentiment_signal": sentiment_result,
        "sentiment_score": score,
        "thought_process": [f"Sentiment Module (FinBERT): {sentiment_result}"]
    }


# ─── Node 4: Validation — Conflict Detection ───

def validation_node(state: AgentState):
    """
    Checks if Vision and Sentiment signals contradict each other.
    If they conflict, flags for Deep Search Agent activation.
    """
    logger.info("--- Validation: Checking signal alignment ---")
    
    vision_signal = state.get('vision_signal', '')
    sentiment_signal = state.get('sentiment_signal', '')
    
    conflict = detect_conflict(vision_signal, sentiment_signal)
    
    if conflict:
        msg = (
            "⚡ CONFLICT DETECTED: Vision and Sentiment disagree! "
            "Activating Deep Search Agent to break the tie..."
        )
    else:
        msg = "✅ Signals aligned — proceeding directly to Memory and Strategy."
    
    return {
        "signals_conflict": conflict,
        "thought_process": [f"Validation: {msg}"]
    }


# ─── Node 5: Deep Search Agent (conditional) ───

def deep_search_node(state: AgentState):
    """
    Triggered only when ViT and FinBERT produce conflicting signals.
    Scrapes additional data and uses Gemini to reconcile.
    """
    logger.info("--- Deep Search Agent: Resolving conflict ---")
    
    ticker = state['ticker']
    vision_signal = state.get('vision_signal', '')
    sentiment_signal = state.get('sentiment_signal', '')
    headlines = state.get('news_headlines', [])
    
    result = deep_search_resolve(ticker, vision_signal, sentiment_signal, headlines)
    
    return {
        "deep_search_result": result,
        "thought_process": [f"Deep Search Agent: {result[:200]}..."]
    }


# ─── Node 6: Memory Analyst ───

def memory_node(state: AgentState):
    """
    Retrieves past market patterns from ChromaDB.
    Queries: 'Have I seen this before? What happened last time?'
    """
    logger.info("--- Agent 3: Memory Analyst (ChromaDB) ---")
    
    current_context = f"{state.get('vision_signal', '')} | {state.get('sentiment_signal', '')}"
    
    # Recall similar regimes
    past_regimes = market_memory.recall_similar_regimes(current_context, n_results=1)
    memory_insight = past_regimes[0] if past_regimes else "No relevant past memory."
    
    # Query past patterns for self-correction
    past_patterns = market_memory.query_past_patterns(current_context, n_results=3)
    
    self_correction = ""
    if past_patterns:
        correct_count = sum(1 for p in past_patterns if p.get('was_correct'))
        total = len(past_patterns)
        if total > 0:
            self_correction = (
                f" | Self-Correction: Found {total} similar past patterns, "
                f"{correct_count}/{total} were profitable."
            )
    
    # Get overall self-correction insight
    insight = market_memory.get_self_correction_insight(state.get('ticker'))
    
    return {
        "memory_signal": memory_insight + self_correction,
        "thought_process": [
            f"Memory Module (ChromaDB): {memory_insight[:100]}...",
            f"Self-Correction: {insight}"
        ]
    }


# ─── Node 7: Lead Strategist ───

def strategist_node(state: AgentState):
    """
    Lead Strategist Agent powered by Google Gemini API.
    Reconciles vision, sentiment, memory, and deep search (if available) to construct a final thesis.
    """
    logger.info("--- Agent 4: Lead Strategist (Gemini) ---")
    ticker = state['ticker']
    current_position = state.get('current_position', False)
    sell_target_date = state.get('sell_target_date', 'None')
    
    position_context = (
        f"You CURRENTLY HOLD a position in {ticker}. Target Sell Date: {sell_target_date}."
        if current_position else
        f"You DO NOT hold any position in {ticker} currently."
    )
    
    # Include deep search result if conflict was detected
    deep_search_context = ""
    if state.get('signals_conflict', False) and state.get('deep_search_result'):
        deep_search_context = f"""
    5. CONFLICT RESOLUTION (Deep Search Agent): {state['deep_search_result']}
    NOTE: The Vision and Sentiment agents DISAGREED. The Deep Search Agent was consulted
    and produced the above analysis. Weight this heavily in your decision.
    """
    
    prompt = f"""
    You are the Lead Strategist Agent for a Hedge Fund.
    You must evaluate the following inputs for the ticker {ticker}.
    
    {position_context}
    
    1. Vision Input (ViT Chart Pattern Analysis): {state.get('vision_signal', 'N/A')}
    2. Semantic Sentiment (FinBERT News Analysis): {state.get('sentiment_signal', 'N/A')}
    3. Memory Input (Historical Pattern Context): {state.get('memory_signal', 'N/A')}
    4. Vision Confidence: {state.get('vision_confidence', 0.0):.2f}
    {deep_search_context}
    
    If you DO NOT hold a position:
    Decide whether to BUY or HOLD. If BUY, specify the number of days to hold before re-evaluating risk to sell.
    
    If you CURRENTLY HOLD a position:
    Decide whether to SELL now or EXTEND the hold mathematically. If EXTEND, specify the number of days to extend.
    
    Output exactly three lines:
    Line 1: DECISION: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL / EXTEND]
    Line 2: CONFIDENCE: [A number from 0.0 to 1.0]
    Line 3: TARGET_HOLD_DAYS: [integer]
    """
    
    # Try models in order of preference with fallback
    models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    response = None
    
    for model_name in models_to_try:
        try:
            logger.info(f"Strategist: Trying model {model_name}...")
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
            response = llm.invoke([
                SystemMessage(content="You are a financial quantitative strategist algorithm."),
                HumanMessage(content=prompt)
            ])
            logger.info(f"Strategist: Successfully used {model_name}")
            break
        except Exception as e:
            logger.warning(f"Strategist: {model_name} failed ({e}), trying next model...")
            continue
    
    if response is not None:
        content = response.content.strip().split('\n')
        
        # Parse response — find lines containing DECISION/CONFIDENCE/TARGET
        decision_line = "DECISION: HOLD"
        confidence_line = "CONFIDENCE: 0.0"
        hold_days_line = "TARGET_HOLD_DAYS: 0"
        
        for line in content:
            line_stripped = line.strip()
            if 'DECISION:' in line_stripped:
                decision_line = line_stripped
            elif 'CONFIDENCE:' in line_stripped:
                confidence_line = line_stripped
            elif 'TARGET_HOLD_DAYS:' in line_stripped:
                hold_days_line = line_stripped
        
        decision = decision_line.split("DECISION:")[-1].strip()
        
        try:
            confidence = float(confidence_line.split("CONFIDENCE:")[-1].strip())
        except ValueError:
            confidence = 0.5
            
        try:
            hold_days = int(hold_days_line.split("TARGET_HOLD_DAYS:")[-1].strip())
        except ValueError:
            hold_days = 0
    else:
        logger.error("Strategist: All Gemini models failed!")
        decision = "HOLD"
        confidence = 0.0
        hold_days = 0

    # Store verdict in memory for self-correction tracking
    try:
        verdict_id = market_memory.store_verdict(
            ticker=ticker,
            vision_signal=state.get('vision_signal', ''),
            sentiment_signal=state.get('sentiment_signal', ''),
            decision=decision,
            confidence=confidence,
            deep_search_used=state.get('signals_conflict', False),
            deep_search_result=state.get('deep_search_result', '')
        )
        logger.info(f"Verdict stored with ID: {verdict_id}")
    except Exception as e:
        logger.warning(f"Failed to store verdict in memory: {e}")

    return {
        "strategist_decision": decision,
        "strategist_confidence": confidence,
        "thought_process": [
            f"Strategist Module (Gemini): Decision='{decision}' Confidence={confidence:.2f} Hold={hold_days} days."
        ]
    }


# ─── Node 8: Risk / Safety Circuit ───

def risk_execution_node(state: AgentState):
    """
    Deterministic Safety Circuit using Kelly Criterion + Circuit Breaker.
    Receives Strategy's request and validates mathematically.
    """
    logger.info("--- Safety Circuit: Evaluating Kelly Criterion + Circuit Breaker ---")
    
    decision = state.get('strategist_decision', 'HOLD')
    confidence = state.get('strategist_confidence', 0.0)
    
    # Check circuit breaker first
    session_pnl = state.get('session_pnl', 0.0)
    initial_capital = state.get('initial_capital', 10000.0)
    
    cb_result = circuit_breaker.check(session_pnl, initial_capital)
    
    if cb_result['triggered']:
        # Generate post-mortem
        post_mortem = circuit_breaker.generate_post_mortem(session_pnl, initial_capital)
        
        return {
            "final_execution_status": "HALTED_CIRCUIT_BREAKER",
            "final_quantity": 0.0,
            "circuit_breaker_triggered": True,
            "post_mortem_report": post_mortem,
            "thought_process": [
                f"🚨 CIRCUIT BREAKER: {cb_result['message']}",
                "Risk Guardrail: ALL TRADING HALTED. Post-mortem report generated."
            ]
        }
    
    # Standard risk evaluation
    if "BUY" not in decision.upper() and "EXTEND" not in decision.upper() and "SELL" not in decision.upper():
        return {
            "final_execution_status": "REJECTED_BY_STRATEGY",
            "final_quantity": 0.0,
            "circuit_breaker_triggered": False,
            "thought_process": [f"Risk Guardrail: Ignored (Decision was {decision}, no execution required)."]
        }
    
    mock_capital = initial_capital if initial_capital > 0 else 10000.0
    requested_amount = 1000.0
    
    if "SELL" in decision.upper():
        status = "APPROVED_SELL"
        amount = requested_amount
        reason = "Lead Strategist requested SELL based on timeline/risk."
    else:
        validation = risk_guardrail.validate_trade(
            confidence_score=confidence,
            current_capital=mock_capital,
            requested_amount=requested_amount
        )
        approved = validation['approved']
        status = "APPROVED_BUY_OR_EXTEND" if approved else "REJECTED_RISK_LIMIT"
        amount = requested_amount if approved else 0.0
        reason = validation['reason']
    
    # Record trade for circuit breaker tracking
    circuit_breaker.record_trade(
        ticker=state.get('ticker', 'UNKNOWN'),
        decision=decision,
        amount=amount,
        pnl=0.0  # Actual PnL determined later
    )
    
    return {
        "final_execution_status": status,
        "final_quantity": amount,
        "circuit_breaker_triggered": False,
        "thought_process": [f"Risk Guardrail (Kelly + CB): {reason} => Status: {status} | CB: {cb_result['message']}"]
    }
