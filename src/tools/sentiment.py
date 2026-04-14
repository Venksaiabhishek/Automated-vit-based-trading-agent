import logging
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Using an LLM to evaluate sentiment is preferred, but for now we'll 
# build the text scraper the LLM can consume, or a mock tool.
# We'll use a standard financial news fetcher.

def fetch_financial_news(ticker: str, limit: int = 5) -> list[str]:
    """Scrapes recent financial news headlines for a given ticker."""
    url = f"https://www.google.com/search?q={ticker}+stock+financial+news&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Google News structure (often changes, but works as a simple fallback)
        headlines = [g.text for g in soup.find_all('div', attrs={'class': 'BNeawe vvjw7b Title79u'})]
        
        if not headlines:
            logger.warning(f"No headlines found for {ticker}")
            return []
            
        return headlines[:limit]
        
    except Exception as e:
        logger.error(f"Failed to fetch news for {ticker}: {e}")
        return []

def analyze_market_sentiment(ticker: str) -> str:
    """
    Scrapes recent financial news for a given ticker, returning the raw headlines.
    This information can then be processed by the main Cognitive Core for Semantic Sentiment Analysis.
    Use this tool to get fundamental context for a stock.
    """
    headlines = fetch_financial_news(ticker)
    
    if not headlines:
        return f"Sentiment Analysis: No recent news found for {ticker}."
        
    formatted_news = "\n- ".join(headlines)
    return (
        f"Semantic Sentiment News Data for {ticker}:\n"
        f"- {formatted_news}\n"
        "Please analyze the underlying fear, greed, or macro-trends from these headlines."
    )
