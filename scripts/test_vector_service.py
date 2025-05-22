"""
Test script for the vector service
"""
import sys
import os
import asyncio
import uuid

# Add the parent directory to the path so we can import the app package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.embedding_service import EmbeddingService
from app.services.vector_service import init_vector_db, VectorService

async def test_vector_service():
    """Test the vector service functionality with Qdrant"""
    # Initialize the embedding service
    embedding_service = EmbeddingService()
    
    # Initialize the vector service
    client = init_vector_db(
        url="http://localhost:6333",
        collection_name="test_collection",
        vector_size=384
    )
    vector_service = VectorService(client, "test_collection")
    
    # Get initial collection info
    collection_info = await vector_service.get_collection_info()
    print(f"Initial collection info: {collection_info}")
    
    # Create test data
    texts = [
        "Python is a versatile programming language used for web development, data analysis, AI, and more.",
        "FastAPI is a modern, high-performance web framework for building APIs with Python.",
        "Vector databases like Qdrant are optimized for similarity search and nearest neighbor queries.",
        "Sentence transformers convert text into fixed-size vector embeddings.",
        "Redis is an in-memory data structure store used for caching, message brokering, and more."
    ]
    
    # Generate embeddings
    embeddings = embedding_service.generate_embeddings(texts)
    
    # Create IDs and payloads
    ids = [str(uuid.uuid4()) for _ in range(len(texts))]
    payloads = [
        {
            "text": text,
            "source_id": f"source_{i}",
            "source_type": "test",
            "metadata": {"index": i}
        }
        for i, text in enumerate(texts)
    ]
    
    # Store vectors
    print(f"Storing {len(embeddings)} vectors...")
    success = await vector_service.store_vectors(
        vectors=embeddings,
        ids=ids,
        payloads=payloads
    )
    print(f"Storage success: {success}")
    
    # Get updated collection info
    collection_info = await vector_service.get_collection_info()
    print(f"Updated collection info: {collection_info}")
    
    # Test search
    query_text = "Python web framework for APIs"
    print(f"Searching for similar texts to: '{query_text}'")
    
    query_embedding = embedding_service.generate_embedding(query_text)
    search_results = await vector_service.search_similar(
        query_vector=query_embedding,
        top_k=3,
        score_threshold=0.5
    )
    
    # Print search results
    print(f"Found {len(search_results)} similar texts:")
    for i, result in enumerate(search_results):
        print(f"{i+1}. Score: {result['score']:.4f}")
        print(f"   Text: {result['payload']['text']}")
        print()

if __name__ == "__main__":
    asyncio.run(test_vector_service())