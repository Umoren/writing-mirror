"""
Integration script to add Gmail chunks to your existing Qdrant vector database
This extends your existing vector setup to handle multiple sources using your existing services
"""

import asyncio
import os
from typing import List, Dict, Any
from app.services.multi_source_processor import MultiSourceProcessor
from app.services.vector_service import VectorService, init_vector_db
from app.services.embedding_service import EmbeddingService

class VectorDatabaseIntegrator:
    def __init__(self, collection_name: str = None):
        # Use your existing collection name from environment
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "writing_samples")
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        
        # Initialize with your existing setup
        self.client = init_vector_db(url=qdrant_url, collection_name=self.collection_name)
        self.vector_service = VectorService(client=self.client, collection_name=self.collection_name)
        self.embedding_service = EmbeddingService()  # Your existing model
        
    async def add_gmail_to_existing_collection(self):
        """Add Gmail chunks to your existing Qdrant collection using your existing services"""
        
        print(f"🔄 Adding Gmail data to existing collection: {self.collection_name}")
        
        # Process Gmail chunks using your existing DocumentProcessor format
        processor = MultiSourceProcessor()
        gmail_chunks = await processor._process_gmail()
        
        if not gmail_chunks:
            print("⚠️  No Gmail chunks to add")
            return
        
        print(f"📧 Processing {len(gmail_chunks)} Gmail chunks...")
        
        # Use your existing vector storage format
        texts = [chunk["text"] for chunk in gmail_chunks]
        ids = [chunk["id"] for chunk in gmail_chunks]
        
        # Create payloads for each chunk with metadata and text (your existing format)
        payloads = []
        for chunk in gmail_chunks:
            # Include both metadata and the actual text in the payload (matches your IntegrationService)
            payload = {
                **chunk["metadata"],
                "text": chunk["text"]  # Include the text in payload for retrieval
            }
            payloads.append(payload)

        # Convert texts to vectors using your existing EmbeddingService
        vectors = self.embedding_service.generate_embeddings(texts)

        # Store in vector database using your existing VectorService
        await self.vector_service.store_vectors(
            vectors=vectors,
            ids=ids,
            payloads=payloads
        )
        
        print(f"🎉 Successfully added {len(gmail_chunks)} Gmail chunks to vector database")
        
        # Show collection stats using your existing method
        collection_info = await self.vector_service.get_collection_info()
        print(f"📊 Total vectors in collection: {collection_info['vectors_count']}")
        
        return len(gmail_chunks)
    
    async def test_mixed_search(self, query: str, limit: int = 5):
        """Test search across both Notion and Gmail sources using your existing VectorService"""
        
        print(f"🔍 Testing mixed search for: '{query}'")
        
        # Generate query embedding using your existing EmbeddingService
        query_vector = self.embedding_service.generate_embedding(query)
        
        # Search using your existing VectorService
        search_results = await self.vector_service.search_similar(
            query_vector=query_vector,
            top_k=limit,
            score_threshold=0.7
        )
        
        print(f"📋 Found {len(search_results)} results:")
        
        for i, result in enumerate(search_results, 1):
            source = result['payload'].get('source', 'unknown')
            text_preview = result['payload'].get('text', '')[:100]
            score = result['score']
            
            print(f"\n{i}. Source: {source} (Score: {score:.3f})")
            print(f"   Text: {text_preview}...")
            
            # Show source-specific metadata
            if source == 'gmail':
                title = result['payload'].get('title', 'N/A')
                print(f"   Subject: {title}")
            elif source == 'notion':
                title = result['payload'].get('title', 'N/A')
                print(f"   Page: {title}")
        
        return search_results
    
    async def get_collection_stats(self):
        """Get statistics about the collection by source using your existing VectorService"""
        
        # Use your existing method
        collection_info = await self.vector_service.get_collection_info()
        
        print(f"📊 Collection Statistics:")
        print(f"   Total vectors: {collection_info['vectors_count']}")
        print(f"   Total points: {collection_info['points_count']}")
        print(f"   Status: {collection_info['status']}")
        
        return collection_info
    
    async def setup_incremental_updates(self):
        """Set up system for incremental Gmail updates using your existing services"""
        
        print("⚙️  Setting up incremental update system...")
        
        processor = MultiSourceProcessor()
        
        # Get recent Gmail chunks (last 24 hours)
        recent_chunks = await processor.process_incremental_gmail(hours_back=24)
        
        if recent_chunks:
            print(f"📬 Found {len(recent_chunks)} recent Gmail chunks")
            
            # Use your existing vector storage format
            texts = [chunk["text"] for chunk in recent_chunks]
            ids = [chunk["id"] for chunk in recent_chunks]
            
            # Create payloads for each chunk with metadata and text
            payloads = []
            for chunk in recent_chunks:
                payload = {
                    **chunk["metadata"],
                    "text": chunk["text"]
                }
                payloads.append(payload)

            # Convert texts to vectors
            vectors = self.embedding_service.generate_embeddings(texts)

            # Store using your existing VectorService
            await self.vector_service.store_vectors(
                vectors=vectors,
                ids=ids,
                payloads=payloads
            )
            print(f"✅ Added {len(recent_chunks)} recent chunks")
        else:
            print("📭 No recent Gmail updates found")

# Usage example - integrates with your existing system
async def integrate_gmail_with_existing_system():
    """Main integration function that works with your existing vector database"""
    
    print("🚀 Integrating Gmail with existing vector database using your existing services...\n")
    
    integrator = VectorDatabaseIntegrator()
    
    # Step 1: Add Gmail data to existing collection
    gmail_count = await integrator.add_gmail_to_existing_collection()
    
    # Step 2: Test mixed search
    test_queries = [
        "frontend developer job",
        "team building",
        "login notification"
    ]
    
    for query in test_queries:
        print(f"\n" + "="*50)
        await integrator.test_mixed_search(query, limit=3)
    
    # Step 3: Show final stats
    print(f"\n" + "="*50)
    await integrator.get_collection_stats()
    
    print(f"\n✅ Integration complete!")
    print(f"📈 Your search API now returns results from both Notion AND Gmail")
    print(f"📧 Added {gmail_count} Gmail chunks to your existing collection")
    
    return gmail_count

if __name__ == "__main__":
    asyncio.run(integrate_gmail_with_existing_system())