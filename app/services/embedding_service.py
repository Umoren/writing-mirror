"""
Embedding service using sentence-transformers
"""
import logging
from typing import List, Union, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and cache the sentence-transformers embedding model
    
    Args:
        model_name: Name of the model to load (default: all-MiniLM-L6-v2)
        
    Returns:
        SentenceTransformer: The loaded model
    """
    logger.info(f"Loading embedding model: {model_name}")
    
    try:
        model = SentenceTransformer(model_name)
        logger.info(f"Model {model_name} loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model = get_embedding_model(model_name)
        logger.info(f"Embedding service initialized with model: {model_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text
        
        Args:
            text: The text to embed
            
        Returns:
            List[float]: The embedding vector
        """
        if not text or not text.strip():
            logger.warning("Attempted to generate embedding for empty text")
            # Return a zero vector with the correct dimension
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            # Convert numpy array to Python list for JSON serialization
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            logger.warning("Attempted to generate embeddings for empty text list")
            return []
        
        # Filter out empty texts
        filtered_texts = [text for text in texts if text and text.strip()]
        
        if not filtered_texts:
            logger.warning("All texts were empty or whitespace")
            return []
        
        try:
            # Generate embeddings in batch
            embeddings = self.model.encode(filtered_texts, convert_to_numpy=True)
            # Convert numpy arrays to Python lists for JSON serialization
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings in batch: {e}")
            raise
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        if not text1 or not text2:
            logger.warning("Attempted to compute similarity with empty text")
            return 0.0
        
        try:
            # Encode both texts
            embedding1 = self.model.encode(text1, convert_to_numpy=True)
            embedding2 = self.model.encode(text2, convert_to_numpy=True)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            raise