"""
Sentiment Analysis Tool — FinBERT-powered
Uses ProsusAI/finbert for financial sentiment classification.
Integrates the inference pipeline from finbert-branch-output.ipynb (Cells 4-5).
"""
import os
import logging
import numpy as np
import torch
from typing import Optional

logger = logging.getLogger(__name__)

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Global model state (lazy loaded) ---
_tokenizer = None
_finbert_model = None
_model_loaded = False

FINBERT_LABELS = ['negative', 'neutral', 'positive']


def _load_finbert():
    """Lazy-load the FinBERT model and tokenizer."""
    global _tokenizer, _finbert_model, _model_loaded
    
    if _model_loaded:
        return
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_name = "ProsusAI/finbert"
        logger.info(f"Loading FinBERT model from {model_name}...")
        
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        _finbert_model.eval()
        
        _model_loaded = True
        logger.info(f"FinBERT loaded successfully on {DEVICE}")
        
    except ImportError:
        logger.error("transformers library not installed. Run: pip install transformers")
        _model_loaded = True  # Prevent repeated attempts
    except Exception as e:
        logger.error(f"Failed to load FinBERT: {e}")
        _model_loaded = True


def finbert_analyze(texts: list[str], batch_size: int = 32) -> dict:
    """
    Run FinBERT inference on a list of texts.
    Based on finbert-branch-output.ipynb Cell 4.
    
    Returns dict with:
        - 'probs': np.ndarray of shape (N, 3) — [prob_neg, prob_neu, prob_pos]
        - 'labels': list of str — predicted labels per text
        - 'scores': list of float — composite scores (pos - neg)
    """
    _load_finbert()
    
    if _finbert_model is None or _tokenizer is None:
        logger.error("FinBERT model not available")
        return {
            'probs': np.zeros((len(texts), 3)),
            'labels': ['neutral'] * len(texts),
            'scores': [0.0] * len(texts)
        }
    
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = _tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors='pt'
        ).to(DEVICE)
        
        with torch.no_grad():
            logits = _finbert_model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    
    probs_array = np.concatenate(all_probs, axis=0)
    labels = [FINBERT_LABELS[i] for i in probs_array.argmax(axis=1)]
    scores = (probs_array[:, 2] - probs_array[:, 0]).tolist()  # pos - neg
    
    return {
        'probs': probs_array,
        'labels': labels,
        'scores': scores
    }


def aggregate_sentiment(labels: list[str], scores: list[float]) -> dict:
    """
    Aggregate multiple article sentiments into a summary.
    Based on finbert-branch-output.ipynb Cell 5.
    
    Returns dict with:
        - 'overall_score': weighted mean sentiment score
        - 'article_count': number of articles
        - 'pos_ratio': fraction of positive articles
        - 'neg_ratio': fraction of negative articles
        - 'score_std': standard deviation (high = conflicting signals)
        - 'overall_label': 'Bullish' | 'Bearish' | 'Neutral'
    """
    if not scores:
        return {
            'overall_score': 0.0,
            'article_count': 0,
            'pos_ratio': 0.0,
            'neg_ratio': 0.0,
            'score_std': 0.0,
            'overall_label': 'Neutral'
        }
    
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    
    overall_score = float(np.mean(scores_arr))
    pos_ratio = float(np.mean(labels_arr == 'positive'))
    neg_ratio = float(np.mean(labels_arr == 'negative'))
    score_std = float(np.std(scores_arr)) if len(scores_arr) > 1 else 0.0
    
    if overall_score > 0.15:
        overall_label = 'Bullish'
    elif overall_score < -0.15:
        overall_label = 'Bearish'
    else:
        overall_label = 'Neutral'
    
    return {
        'overall_score': overall_score,
        'article_count': len(scores),
        'pos_ratio': pos_ratio,
        'neg_ratio': neg_ratio,
        'score_std': score_std,
        'overall_label': overall_label
    }


def analyze_market_sentiment(ticker: str) -> str:
    """
    Full sentiment analysis pipeline: fetch news → run FinBERT → aggregate → return structured verdict.
    This is the main entry point called by the agent nodes.
    """
    # Import here to avoid circular imports
    from ..data.ingestion import fetch_news_headlines
    
    articles = fetch_news_headlines(ticker, limit=100)
    
    if not articles:
        return f"Sentiment Agent: Neutral — No recent news found for {ticker} (score: 0.00, 0 articles)"
    
    # Combine headline + summary for richer context (from finbert notebook)
    texts = []
    for a in articles:
        headline = a.get('headline', '')
        summary = a.get('summary', '')[:100] if a.get('summary') else ''
        combined = f"{headline}. {summary}" if summary else headline
        texts.append(combined)
    
    # Run FinBERT
    result = finbert_analyze(texts)
    
    # Aggregate
    agg = aggregate_sentiment(result['labels'], result['scores'])
    
    return (
        f"Sentiment Agent: {agg['overall_label']} "
        f"(score: {agg['overall_score']:.2f}, "
        f"{agg['article_count']} articles, "
        f"{agg['pos_ratio']*100:.1f}% positive, "
        f"{agg['neg_ratio']*100:.1f}% negative, "
        f"std: {agg['score_std']:.2f})"
    )


def get_sentiment_parsed(result_str: str) -> tuple[str, float]:
    """
    Parses the sentiment result string into (signal, score).
    Returns ('Bullish'|'Bearish'|'Neutral', score_float).
    """
    signal = 'Neutral'
    score = 0.0
    
    if 'Bullish' in result_str:
        signal = 'Bullish'
    elif 'Bearish' in result_str:
        signal = 'Bearish'
    
    try:
        score_str = result_str.split('score:')[1].strip().split(',')[0].strip()
        score = float(score_str)
    except (IndexError, ValueError):
        pass
    
    return signal, score
