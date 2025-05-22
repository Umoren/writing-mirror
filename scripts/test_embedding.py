"""
Test script for the embedding service
"""
import sys
import os

# Add the parent directory to the path so we can import the app package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.embedding_service import EmbeddingService

def test_embedding_service():
    """Test the embedding service functionality"""
    # Initialize the service
    service = EmbeddingService()
    
    # Test single text embedding
    text = "This is a test sentence for embedding."
    embedding = service.generate_embedding(text)
    
    print(f"Generated embedding for: '{text}'")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First few values: {embedding[:5]}...")
    
    # Test multiple text embeddings
    texts = [
        "This is the first test sentence.",
        "Here is another completely different sentence.",
        "And a third one just to be sure."
    ]
    
    embeddings = service.generate_embeddings(texts)
    
    print(f"\nGenerated {len(embeddings)} embeddings for {len(texts)} texts")
    print(f"Each embedding dimension: {len(embeddings[0])}")
    
    # Test similarity
    text1 = "I love machine learning and natural language processing."
    text2 = "Deep learning and NLP are fascinating fields of study."
    text3 = "The stock market went up by two percent today."
    
    sim1_2 = service.compute_similarity(text1, text2)
    sim1_3 = service.compute_similarity(text1, text3)
    
    print(f"\nSimilarity between related texts: {sim1_2:.4f}")
    print(f"Similarity between unrelated texts: {sim1_3:.4f}")

if __name__ == "__main__":
    test_embedding_service()