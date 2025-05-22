"""
Test script for the integration service
"""
import sys
import os
import asyncio
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the app package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

from app.services.notion_service import NotionService
from app.services.vector_service import VectorService, init_vector_db
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.integration_service import IntegrationService

async def test_integration():
    """Test the integration service functionality"""
    # Get API keys and configuration from environment
    notion_api_key = os.getenv("NOTION_API_KEY")
    notion_database_id = os.getenv("NOTION_DATABASE_ID")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "writing_samples")
    
    if not notion_api_key or not notion_database_id:
        print("ERROR: NOTION_API_KEY and NOTION_DATABASE_ID environment variables must be set.")
        return
    
    if not qdrant_url:
        print("ERROR: QDRANT_URL environment variable must be set.")
        return
    
    # Initialize services
    notion_service = NotionService(api_key=notion_api_key, database_id=notion_database_id)

    # Initialize Qdrant client first
    qdrant_client = init_vector_db(url=qdrant_url, collection_name=collection_name)

    # Then initialize the VectorService with the client
    vector_service = VectorService(client=qdrant_client, collection_name=collection_name)

    document_processor = DocumentProcessor(chunk_size=512, chunk_overlap=128)
    embedding_service = EmbeddingService()  # Use default model
    
    integration_service = IntegrationService(
        notion_service=notion_service,
        vector_service=vector_service,
        document_processor=document_processor,
        embedding_service=embedding_service,
        state_file_path="data/sync_state.json"
    )
    
    # Sync documents
    print("Syncing documents from Notion to Qdrant...")
    stats = await integration_service.sync_documents(force_full_sync=True)
    
    print(f"Sync stats:")
    for key, value in stats.items():
        print(f"- {key}: {value}")
    
    # Test search
    search_queries = [
        "Docker and containerization",
        "API performance optimization",
        "node.js development best practices",
        "data security and privacy",
        "microservices architecture"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: {query}")
        results = await integration_service.search_similar_texts(query=query, limit=3)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"{i+1}. Score: {result['score']:.4f}")
                print(f"   Title: {result['metadata'].get('title')}")
                print(f"   Preview: {result['text'][:200]}...")
        else:
            print("No results found.")

if __name__ == "__main__":
    asyncio.run(test_integration())