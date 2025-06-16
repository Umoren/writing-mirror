#!/usr/bin/env python3
"""
Fix the vector database by clearing and rebuilding with proper embeddings
"""

import asyncio
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.vector_service import VectorService, init_vector_db
from app.services.embedding_service import EmbeddingService
from app.services.multi_source_processor import MultiSourceProcessor

async def fix_vector_database():
    """Clear and rebuild the vector database with proper embeddings"""
    
    print("🔧 Fixing Vector Database...")
    
    # Initialize services
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "writing_samples")
    
    qdrant_client = init_vector_db(url=qdrant_url, collection_name=collection_name)
    vector_service = VectorService(client=qdrant_client, collection_name=collection_name)
    embedding_service = EmbeddingService()
    
    # Step 1: Clear the collection
    print("🗑️  Clearing existing collection...")
    try:
        qdrant_client.delete_collection(collection_name)
        print("✅ Collection cleared")
    except Exception as e:
        print(f"Note: {e}")
    
    # Step 2: Recreate collection
    print("🆕 Recreating collection...")
    qdrant_client = init_vector_db(url=qdrant_url, collection_name=collection_name)
    vector_service = VectorService(client=qdrant_client, collection_name=collection_name)
    
    # Step 3: Process Gmail data
    print("📧 Processing Gmail data...")
    processor = MultiSourceProcessor()
    gmail_chunks = await processor._process_gmail()
    
    if not gmail_chunks:
        print("❌ No Gmail chunks found")
        return
    
    print(f"📊 Found {len(gmail_chunks)} Gmail chunks")
    
    # Step 4: Generate embeddings in batches (to avoid memory issues)
    batch_size = 50
    total_stored = 0
    
    for i in range(0, len(gmail_chunks), batch_size):
        batch = gmail_chunks[i:i + batch_size]
        print(f"🔄 Processing batch {i//batch_size + 1}/{(len(gmail_chunks) + batch_size - 1)//batch_size}")
        
        # Extract texts
        texts = [chunk["text"] for chunk in batch]
        
        # Generate embeddings
        print(f"  Generating {len(texts)} embeddings...")
        embeddings = embedding_service.generate_embeddings(texts)
        
        # Verify embeddings
        if not embeddings or len(embeddings) != len(texts):
            print(f"  ❌ Embedding generation failed for batch {i//batch_size + 1}")
            continue
            
        # Check embedding dimensions
        if embeddings and len(embeddings[0]) == 0:
            print(f"  ❌ Generated embeddings have zero dimension")
            continue
            
        print(f"  ✅ Generated embeddings with dimension {len(embeddings[0])}")
        
        # Prepare data for storage
        ids = [chunk["id"] for chunk in batch]
        payloads = []
        
        for chunk in batch:
            payload = {
                **chunk["metadata"],
                "text": chunk["text"]  # Include text in payload
            }
            payloads.append(payload)
        
        # Store vectors
        print(f"  💾 Storing {len(embeddings)} vectors...")
        success = await vector_service.store_vectors(
            vectors=embeddings,
            ids=ids,
            payloads=payloads
        )
        
        if success:
            total_stored += len(embeddings)
            print(f"  ✅ Stored batch successfully")
        else:
            print(f"  ❌ Failed to store batch")
    
    print(f"\n🎉 Rebuild complete!")
    print(f"📊 Total vectors stored: {total_stored}")
    
    # Step 5: Verify the fix
    print("\n🔍 Verifying fix...")
    
    collection_info = await vector_service.get_collection_info()
    print(f"Collection points: {collection_info.get('points_count', 0)}")
    
    # Test search
    query = "frontend developer"
    query_vector = embedding_service.generate_embedding(query)
    
    results = await vector_service.search_similar(
        query_vector=query_vector,
        top_k=3,
        score_threshold=0.1
    )
    
    print(f"Test search for '{query}': {len(results)} results")
    
    if results:
        print("✅ Vector database is working!")
        for i, result in enumerate(results[:2]):
            print(f"  {i+1}. Score: {result['score']:.3f}")
            print(f"     Title: {result['payload'].get('title', 'N/A')[:50]}...")
    else:
        print("❌ Search still not working")

if __name__ == "__main__":
    asyncio.run(fix_vector_database())