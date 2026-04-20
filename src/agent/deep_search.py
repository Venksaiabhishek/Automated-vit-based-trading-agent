"""
Deep Search Agent — Contradiction Resolution Module
When the Vision Agent (ViT) and Sentiment Agent (FinBERT) produce conflicting signals,
this agent scrapes additional data and uses Gemini LLM to break the tie.
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Bridge GEMINI_API_KEY → GOOGLE_API_KEY for langchain-google-genai
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def detect_conflict(vision_signal: str, sentiment_signal: str) -> bool:
    """
    Detects if the vision and sentiment signals contradict each other.
    Returns True if they conflict (one Bullish, other Bearish).
    """
    vision_bullish = 'Bullish' in vision_signal
    vision_bearish = 'Bearish' in vision_signal
    sentiment_bullish = 'Bullish' in sentiment_signal
    sentiment_bearish = 'Bearish' in sentiment_signal
    
    # Conflict: one is bullish and the other is bearish
    if (vision_bullish and sentiment_bearish) or (vision_bearish and sentiment_bullish):
        logger.info("CONFLICT DETECTED: Vision and Sentiment disagree!")
        return True
    
    return False


def deep_search_resolve(
    ticker: str,
    vision_signal: str,
    sentiment_signal: str,
    existing_headlines: list[str] = None
) -> str:
    """
    Deep Search Agent: scrapes additional data and uses Gemini to reconcile
    conflicting Vision and Sentiment signals.
    
    Returns a reasoned tiebreaker verdict.
    """
    logger.info(f"--- Deep Search Agent triggered for {ticker} ---")
    
    # Step 1: Gather additional context
    additional_context = _gather_extra_data(ticker)
    
    # Step 2: Run FinBERT on expanded data if available
    expanded_sentiment = ""
    if additional_context:
        try:
            from ..tools.sentiment import finbert_analyze, aggregate_sentiment
            texts = [a.get('headline', '') for a in additional_context if a.get('headline')]
            if texts:
                result = finbert_analyze(texts)
                agg = aggregate_sentiment(result['labels'], result['scores'])
                expanded_sentiment = (
                    f"Expanded FinBERT Analysis ({agg['article_count']} additional articles): "
                    f"{agg['overall_label']} (score: {agg['overall_score']:.2f}, "
                    f"pos: {agg['pos_ratio']*100:.1f}%, neg: {agg['neg_ratio']*100:.1f}%)"
                )
        except Exception as e:
            logger.warning(f"Expanded FinBERT analysis failed: {e}")
    
    # Step 3: Use Gemini to reconcile
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        prompt = f"""
        You are the Deep Search Analyst for a quantitative hedge fund.
        
        CRITICAL: The Vision Agent and Sentiment Agent are in DISAGREEMENT for ticker {ticker}.
        You have been activated to break the tie.
        
        Vision Agent (Chart Pattern Analysis via ViT): {vision_signal}
        Sentiment Agent (News FinBERT Analysis): {sentiment_signal}
        
        {f'Additional Data Gathered: {expanded_sentiment}' if expanded_sentiment else 'No additional data available.'}
        
        Extra headlines gathered:
        {chr(10).join([f"- {a.get('headline', 'N/A')}" for a in (additional_context or [])[:10]])}
        
        TASK:
        1. Analyze which signal is more likely correct given the additional data
        2. Consider that chart patterns (Vision) reflect institutional activity while news (Sentiment) can be reactive
        3. Provide your tiebreaker verdict
        
        Output exactly:
        TIEBREAKER: [BULLISH / BEARISH / NEUTRAL]
        RATIONALE: [Your reasoning in 2-3 sentences]
        CONFIDENCE: [0.0 to 1.0]
        """
        
        models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
        response = None
        
        for model_name in models_to_try:
            try:
                llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
                response = llm.invoke([
                    SystemMessage(content="You are a quantitative market analyst specializing in signal reconciliation."),
                    HumanMessage(content=prompt)
                ])
                break
            except Exception:
                continue
        
        if response is None:
            raise RuntimeError("All Gemini models unavailable")
        
        result = response.content.strip()
        logger.info(f"Deep Search Resolution: {result[:200]}")
        return f"Deep Search Agent Resolution:\n{result}"
        
    except Exception as e:
        logger.error(f"Deep Search LLM call failed: {e}")
        # Fallback: use expanded sentiment if available, else go with vision (institutional signals)
        if expanded_sentiment:
            return f"Deep Search Agent (Fallback): Using expanded sentiment data. {expanded_sentiment}"
        return f"Deep Search Agent (Fallback): Defaulting to Vision signal — chart patterns reflect institutional movement."


def _gather_extra_data(ticker: str) -> list[dict]:
    """Scrapes additional news for deeper analysis."""
    try:
        from ..data.ingestion import fetch_news_google, fetch_news_alphavantage
        
        # Try Google first (doesn't use API quota)
        articles = fetch_news_google(ticker, limit=15)
        
        # Also try broader AV search if available
        av_articles = fetch_news_alphavantage(ticker, limit=50)
        if av_articles:
            articles.extend(av_articles)
        
        logger.info(f"Deep Search gathered {len(articles)} additional articles for {ticker}")
        return articles
        
    except Exception as e:
        logger.warning(f"Extra data gathering failed: {e}")
        return []
