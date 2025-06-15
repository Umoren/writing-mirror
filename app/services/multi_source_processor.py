"""
Enhanced document processor that handles multiple sources (Notion + Gmail)
This extends your existing processor to handle Gmail alongside Notion
"""

import asyncio
import os
from typing import List, Dict, Any
from app.services.gmail_service import GmailService
from app.services.notion_service import NotionService
from app.services.document_processor import DocumentProcessor

class MultiSourceProcessor:
    def __init__(self):
        self.gmail_service = GmailService()

        # Initialize Notion service with environment variables
        notion_api_key = os.getenv("NOTION_API_KEY")
        notion_database_id = os.getenv("NOTION_DATABASE_ID")

        if notion_api_key and notion_database_id:
            self.notion_service = NotionService(
                api_key=notion_api_key,
                database_id=notion_database_id
            )
        else:
            print("⚠️  NOTION_API_KEY or NOTION_DATABASE_ID not set - Notion integration disabled")
            self.notion_service = None

        # Initialize document processor with same settings as your existing setup
        self.document_processor = DocumentProcessor(chunk_size=512, chunk_overlap=128)

    async def process_all_sources(self) -> List[Dict[str, Any]]:
        """Process documents from all sources and return unified chunks"""
        all_chunks = []

        print("🔄 Processing Gmail...")
        gmail_chunks = await self._process_gmail()
        all_chunks.extend(gmail_chunks)
        print(f"✅ Gmail: {len(gmail_chunks)} chunks")

        if self.notion_service:
            print("🔄 Processing Notion...")
            notion_chunks = await self._process_notion()
            all_chunks.extend(notion_chunks)
            print(f"✅ Notion: {len(notion_chunks)} chunks")
        else:
            print("⚠️  Notion processing skipped (not configured)")

        print(f"🎯 Total chunks from all sources: {len(all_chunks)}")
        return all_chunks
    
    async def _process_gmail(self) -> List[Dict[str, Any]]:
        """Process Gmail emails into chunks using your existing DocumentProcessor"""
        chunks = []

        try:
            # Get recent emails (last 30 days)
            emails = self.gmail_service.get_recent_emails(max_results=50, days_back=30)

            for email in emails:
                # Convert email to your document format
                document = self._gmail_to_document_format(email)

                # Use your existing DocumentProcessor
                email_chunks = self.document_processor.process_document(document)
                chunks.extend(email_chunks)

        except Exception as e:
            print(f"Error processing Gmail: {e}")

        return chunks
    
    def _gmail_to_document_format(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Gmail email to your document format"""
        return {
            "id": email_data['id'],
            "title": email_data['subject'],
            "content": f"From: {email_data['sender']}\nSubject: {email_data['subject']}\n\n{email_data['body']}",
            "created_time": email_data.get('date'),
            "last_edited_time": email_data.get('date'),
            "source": "gmail",  # This will be added to metadata by DocumentProcessor
            # Add Gmail-specific metadata
            "gmail_metadata": {
                "sender": email_data['sender'],
                "thread_id": email_data.get('thread_id'),
                "email_id": email_data['id']
            }
        }

    async def _process_notion(self) -> List[Dict[str, Any]]:
        """Process Notion pages into chunks using your existing services"""
        chunks = []

        if not self.notion_service:
            return chunks

        try:
            # Use your existing Notion processing logic
            documents = await self.notion_service.process_all_documents()

            for document in documents:
                # Use your existing DocumentProcessor
                notion_chunks = self.document_processor.process_document(document)
                chunks.extend(notion_chunks)

        except Exception as e:
            print(f"Error processing Notion: {e}")

        return chunks
    
    def get_source_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get statistics about chunks by source"""
        stats = {}
        for chunk in chunks:
            # Your chunk format has source in metadata
            source = chunk.get('metadata', {}).get('source', 'unknown')
            stats[source] = stats.get(source, 0) + 1
        return stats

    async def process_incremental_gmail(self, hours_back: int = 1) -> List[Dict[str, Any]]:
        """Process only recent Gmail updates (for real-time updates)"""
        chunks = []

        try:
            # Get very recent emails only
            emails = self.gmail_service.get_recent_emails(max_results=10, days_back=1)
            
            # Filter to only emails from last N hours
            import time
            cutoff_time = time.time() - (hours_back * 3600)

            recent_emails = [
                email for email in emails
                if email.get('timestamp', 0) > cutoff_time
            ]

            for email in recent_emails:
                # Convert to document format and process with DocumentProcessor
                document = self._gmail_to_document_format(email)
                email_chunks = self.document_processor.process_document(document)
                chunks.extend(email_chunks)

            print(f"📬 Found {len(recent_emails)} new emails → {len(chunks)} chunks")

        except Exception as e:
            print(f"Error processing incremental Gmail: {e}")

        return chunks


# Example usage function - integrate with your existing IntegrationService
async def update_vector_database():
    """Example of how to integrate this with your existing vector DB using IntegrationService"""
    from app.services.vector_service import VectorService, init_vector_db
    from app.services.embedding_service import EmbeddingService

    processor = MultiSourceProcessor()
    
    # Get all chunks from all sources
    all_chunks = await processor.process_all_sources()
    
    # Show source distribution
    stats = processor.get_source_stats(all_chunks)
    print(f"📊 Source distribution: {stats}")
    
    # Initialize your existing vector services
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "writing_samples")

    qdrant_client = init_vector_db(url=qdrant_url, collection_name=collection_name)
    vector_service = VectorService(client=qdrant_client, collection_name=collection_name)
    embedding_service = EmbeddingService()

    # Process chunks for vector storage (same as your existing IntegrationService)
    if all_chunks:
        texts = [chunk["text"] for chunk in all_chunks]
        ids = [chunk["id"] for chunk in all_chunks]

        # Create payloads for each chunk with metadata and text
        payloads = []
        for chunk in all_chunks:
            # Include both metadata and the actual text in the payload (your existing format)
            payload = {
                **chunk["metadata"],
                "text": chunk["text"]  # Include the text in payload for retrieval
            }
            payloads.append(payload)

        # Convert texts to vectors
        vectors = embedding_service.generate_embeddings(texts)

        # Store in vector database using your existing VectorService
        await vector_service.store_vectors(
            vectors=vectors,
            ids=ids,
            payloads=payloads
        )

        print(f"✅ Added {len(all_chunks)} chunks to vector database")

    return all_chunks


# Test function to verify multi-source processing
async def test_multi_source():
    """Test the multi-source processing with your existing chunk format"""
    processor = MultiSourceProcessor()
    
    print("Testing multi-source document processing...")
    
    # Test Gmail processing
    gmail_chunks = await processor._process_gmail()
    print(f"Gmail chunks: {len(gmail_chunks)}")

    if gmail_chunks:
        sample_chunk = gmail_chunks[0]
        print(f"\nSample Gmail chunk:")
        print(f"ID: {sample_chunk['id']}")
        print(f"Text preview: {sample_chunk['text'][:100]}...")
        print(f"Metadata: {sample_chunk['metadata']}")

    # Test Notion processing if available
    if processor.notion_service:
        notion_chunks = await processor._process_notion()
        print(f"Notion chunks: {len(notion_chunks)}")

        if notion_chunks:
            sample_chunk = notion_chunks[0]
            print(f"\nSample Notion chunk:")
            print(f"ID: {sample_chunk['id']}")
            print(f"Text preview: {sample_chunk['text'][:100]}...")
            print(f"Metadata: {sample_chunk['metadata']}")

    # Test incremental updates
    print(f"\nTesting incremental updates (last 24 hours)...")
    recent_chunks = await processor.process_incremental_gmail(hours_back=24)
    print(f"Recent chunks: {len(recent_chunks)}")

    return gmail_chunks


if __name__ == "__main__":
    # Test the multi-source processor
    asyncio.run(test_multi_source())