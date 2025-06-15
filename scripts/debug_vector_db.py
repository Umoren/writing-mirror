#!/usr/bin/env python3
"""
Debug script to check what's in the vector database and why search isn't working
"""

import asyncio
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.vector_service import VectorService, init_vector_db
from app.services.embedding_service import EmbeddingService

async def debug_vector_database():
    """Debug what's in the vector database"""
    
    print("🔍 Debugging Vector Database...")
    
    # Initialize services
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "writing_samples")
    
    qdrant_client = init_vector_db(url=qdrant_url, collection_name=collection_name)
    vector_service = VectorService(client=qdrant_client, collection_name=collection_name)
    embedding_service = EmbeddingService()
    
    # Get collection info
    collection_info = await vector_service.get_collection_info()
    print(f"📊 Collection Info:")
    print(f"   Name: {collection_name}")
    print(f"   Points: {collection_info.get('points_count', 'Unknown')}")
    print(f"   Vectors: {collection_info.get('vectors_count', 'Unknown')}")
    print(f"   Status: {collection_info.get('status', 'Unknown')}")
    
    # Test different search queries with different thresholds
    test_queries = [
        "frontend developer",
        "job alert", 
        "login",
        "netflix",
        "email"
    ]
    
    thresholds = [0.1, 0.3, 0.5, 0.7]
    
    for query in test_queries[:2]:  # Test first 2 queries
        print(f"\n🔍 Testing query: '{query}'")
        
        # Generate embedding
        query_vector = embedding_service.generate_embedding(query)
        print(f"   Query vector dimension: {len(query_vector)}")
        
        for threshold in thresholds:
            try:
                results = await vector_service.search_similar(
                    query_vector=query_vector,
                    top_k=10,
                    score_threshold=threshold
                )
                
                print(f"   Threshold {threshold}: {len(results)} results")
                
                if results:
                    # Show details of first result
                    result = results[0]
                    payload = result.get('payload', {})
                    print(f"      Best match: score={result.get('score', 0):.3f}")
                    print(f"      Source: {payload.get('source', 'unknown')}")
                    print(f"      Title: {payload.get('title', 'unknown')[:50]}...")
                    
                    # Count sources in results
                    sources = {}
                    for r in results:
                        source = r.get('payload', {}).get('source', 'unknown')
                        sources[source] = sources.get(source, 0) + 1
                    print(f"      Source breakdown: {sources}")
                    break
            except Exception as e:
                print(f"   Threshold {threshold}: Error - {e}")
    
    # Sample some raw data from the database
    print(f"\n📝 Sampling raw data...")
    try:
        # Use Qdrant client directly to scroll through some points
        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=5  # Get 5 sample points
        )
        
        points = scroll_result[0]  # First element is the points list
        
        print(f"   Found {len(points)} sample points:")
        for i, point in enumerate(points):
            payload = point.payload
            print(f"      Point {i+1}:")
            print(f"         ID: {point.id}")
            print(f"         Source: {payload.get('source', 'unknown')}")
            print(f"         Title: {payload.get('title', 'unknown')[:40]}...")
            print(f"         Has text: {'text' in payload}")
            print(f"         Vector dimension: {len(point.vector) if point.vector else 0}")
            
    except Exception as e:
        print(f"   Error sampling data: {e}")

if __name__ == "__main__":
    asyncio.run(debug_vector_database())