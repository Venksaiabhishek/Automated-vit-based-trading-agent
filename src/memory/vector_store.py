"""
Market Memory — Vector Store with Verdict/Outcome Tracking
Manages persistent memory using ChromaDB for:
1. Market regime similarity search (existing)
2. Verdict storage and outcome recording (new)
3. Self-correction insights (new)
"""
import logging
import uuid
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = logging.getLogger(__name__)


class MarketMemory:
    """
    Manages the Persistent Memory Layer using ChromaDB.
    Allows agents to remember past market regimes, verdicts, and outcomes.
    Demonstrates Self-Correction capability.
    """
    def __init__(self, db_dir=".chroma_db", collection_name="market_memory"):
        self.db_dir = db_dir
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.embedding_fn = DefaultEmbeddingFunction()
        
        # Collection for market regime memories
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Collection for verdict/outcome tracking (Self-Correction)
        self.verdicts_collection = self.client.get_or_create_collection(
            name="verdicts",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized Market Memory at {self.db_dir} [Collections: {collection_name}, verdicts]")

    # ─── Original Market Regime Methods ───

    def add_experience(self, regime_id: str, ticker: str, market_context: str, outcome: str):
        """Stores a specific market experience (regime) mapped to its outcome."""
        combined_text = f"Ticker: {ticker} | Context: {market_context} | Outcome: {outcome}"
        
        try:
            self.collection.add(
                documents=[combined_text],
                metadatas=[{"ticker": ticker, "type": "market_regime"}],
                ids=[regime_id]
            )
            logger.info(f"Recorded new market experience for {ticker} (ID: {regime_id})")
        except Exception as e:
            logger.error(f"Failed to add experience to memory: {e}")

    def recall_similar_regimes(self, current_context: str, n_results: int = 3) -> list[str]:
        """Searches for past market regimes semantically matching the current context."""
        try:
            results = self.collection.query(
                query_texts=[current_context],
                n_results=n_results
            )
            retrieved_docs = results.get('documents', [[]])[0]
            if not retrieved_docs:
                return ["No similar past regimes found."]
            return retrieved_docs
        except Exception as e:
            logger.error(f"Failed to recall regimes: {e}")
            return ["Memory Retrieval Error."]

    # ─── NEW: Verdict & Outcome Tracking ───

    def store_verdict(
        self,
        ticker: str,
        vision_signal: str,
        sentiment_signal: str,
        decision: str,
        confidence: float,
        deep_search_used: bool = False,
        deep_search_result: str = ""
    ) -> str:
        """
        Store every trading verdict for later outcome correlation.
        Returns the verdict_id for linking outcomes.
        """
        verdict_id = f"verdict_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        document = (
            f"Ticker: {ticker} | "
            f"Vision: {vision_signal} | "
            f"Sentiment: {sentiment_signal} | "
            f"Decision: {decision} | "
            f"Confidence: {confidence:.2f}"
        )
        
        if deep_search_used:
            document += f" | Deep Search: {deep_search_result[:200]}"
        
        metadata = {
            "ticker": ticker,
            "decision": decision,
            "confidence": confidence,
            "deep_search_used": deep_search_used,
            "timestamp": datetime.now().isoformat(),
            "outcome_recorded": False,
            "outcome": "",
            "pnl": 0.0,
            "was_correct": False
        }
        
        try:
            self.verdicts_collection.add(
                documents=[document],
                metadatas=[metadata],
                ids=[verdict_id]
            )
            logger.info(f"Stored verdict {verdict_id} for {ticker}: {decision}")
            return verdict_id
        except Exception as e:
            logger.error(f"Failed to store verdict: {e}")
            return ""

    def record_outcome(self, verdict_id: str, actual_outcome: str, pnl: float):
        """
        Record what actually happened after a verdict was made.
        Links the outcome to the original verdict for self-correction learning.
        """
        was_correct = pnl > 0
        
        try:
            self.verdicts_collection.update(
                ids=[verdict_id],
                metadatas=[{
                    "outcome_recorded": True,
                    "outcome": actual_outcome,
                    "pnl": pnl,
                    "was_correct": was_correct,
                    "outcome_timestamp": datetime.now().isoformat()
                }]
            )
            emoji = "✅" if was_correct else "❌"
            logger.info(f"{emoji} Recorded outcome for {verdict_id}: {actual_outcome} (PnL: {pnl:+.2f})")
        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")

    def query_past_patterns(self, current_context: str, n_results: int = 5) -> list[dict]:
        """
        'Have I seen this market pattern before? What was the result last time?'
        Returns past verdicts similar to the current context with their outcomes.
        """
        try:
            results = self.verdicts_collection.query(
                query_texts=[current_context],
                n_results=n_results
            )
            
            past_patterns = []
            docs = results.get('documents', [[]])[0]
            metas = results.get('metadatas', [[]])[0]
            
            for doc, meta in zip(docs, metas):
                past_patterns.append({
                    'context': doc,
                    'decision': meta.get('decision', 'Unknown'),
                    'was_correct': meta.get('was_correct', False),
                    'pnl': meta.get('pnl', 0.0),
                    'outcome': meta.get('outcome', 'Not yet recorded'),
                    'timestamp': meta.get('timestamp', '')
                })
            
            return past_patterns
            
        except Exception as e:
            logger.error(f"Failed to query past patterns: {e}")
            return []

    def get_self_correction_insight(self, ticker: str = None) -> str:
        """
        Returns self-correction analytics:
        - Overall accuracy rate
        - Pattern-specific advice
        - When the agent tends to be wrong
        """
        try:
            # Get all verdicts
            where_filter = {"ticker": ticker} if ticker else None
            all_verdicts = self.verdicts_collection.get(
                where=where_filter,
                limit=100
            )
            
            if not all_verdicts['ids']:
                return "Self-Correction: No past verdicts found. Building experience base."
            
            metas = all_verdicts['metadatas']
            total = len(metas)
            recorded = [m for m in metas if m.get('outcome_recorded', False)]
            correct = [m for m in recorded if m.get('was_correct', False)]
            
            if not recorded:
                return f"Self-Correction: {total} verdicts recorded, but no outcomes linked yet."
            
            accuracy = len(correct) / len(recorded) * 100
            total_pnl = sum(m.get('pnl', 0.0) for m in recorded)
            
            # Find worst patterns
            wrong = [m for m in recorded if not m.get('was_correct', False)]
            wrong_decisions = {}
            for m in wrong:
                d = m.get('decision', 'Unknown')
                wrong_decisions[d] = wrong_decisions.get(d, 0) + 1
            
            worst_pattern = max(wrong_decisions, key=wrong_decisions.get) if wrong_decisions else "None"
            
            insight = (
                f"Self-Correction Report:\n"
                f"  Total Verdicts: {total} | With Outcomes: {len(recorded)}\n"
                f"  Accuracy: {accuracy:.1f}% | Total PnL: {total_pnl:+.2f}\n"
                f"  Most Common Wrong Decision: {worst_pattern}\n"
                f"  Advice: {'Reduce position sizing on {worst_pattern} calls' if worst_pattern != 'None' else 'Insufficient data for pattern analysis'}"
            )
            
            return insight
            
        except Exception as e:
            logger.error(f"Self-correction insight failed: {e}")
            return "Self-Correction: Analysis unavailable."

    def get_all_verdicts(self, limit: int = 50) -> list[dict]:
        """Returns all stored verdicts for dashboard display."""
        try:
            results = self.verdicts_collection.get(limit=limit)
            verdicts = []
            for doc, meta, vid in zip(
                results.get('documents', []),
                results.get('metadatas', []),
                results.get('ids', [])
            ):
                verdicts.append({
                    'id': vid,
                    'context': doc,
                    **meta
                })
            return verdicts
        except Exception as e:
            logger.error(f"Failed to get verdicts: {e}")
            return []


# Singleton instance
market_memory = MarketMemory()
