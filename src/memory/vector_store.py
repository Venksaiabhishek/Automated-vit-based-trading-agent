import logging
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = logging.getLogger(__name__)

class MarketMemory:
    """
    Manages the Persistent Memory Layer using ChromaDB.
    This allows agents to remember past market regimes and adjust their 
    current strategy based on what worked or failed previously.
    """
    def __init__(self, db_dir=".chroma_db", collection_name="market_memory"):
        self.db_dir = db_dir
        self.client = chromadb.PersistentClient(path=self.db_dir)
        
        # Use default sentence transformers model for semantic embeddings
        self.embedding_fn = DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized Market Memory at {self.db_dir} [Collection: {collection_name}]")

    def add_experience(self, regime_id: str, ticker: str, market_context: str, outcome: str):
        """
        Stores a specific market experience (regime) mapped to its outcome.
        - market_context: description of macros, chart pattern, etc.
        - outcome: what happened next, and was the agent's trade successful?
        """
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
        """
        Searches the Vector Database for past market regimes that semantically
        match the current context.
        """
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

# Singleton instance to be imported by the agents
market_memory = MarketMemory()
