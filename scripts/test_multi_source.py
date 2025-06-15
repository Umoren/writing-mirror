#!/usr/bin/env python3
"""
Test script for multi-source document processing
This verifies Gmail integration works with your existing pipeline
"""

import sys
import os
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.multi_source_processor import MultiSourceProcessor

async def test_gmail_chunks():
    """Test Gmail chunk processing in detail using existing DocumentProcessor"""
    print("=== Testing Gmail Chunk Processing ===")
    
    processor = MultiSourceProcessor()
    
    # Get Gmail chunks using the new integrated approach
    gmail_chunks = await processor._process_gmail()
    
    print(f"📧 Total Gmail chunks: {len(gmail_chunks)}")
    
    if not gmail_chunks:
        print("⚠️  No Gmail chunks found. Check if you have recent emails.")
        return []
    
    # Analyze chunk quality using your existing chunk format
    print(f"\n📊 Chunk Analysis:")
    
    sources = set(chunk['metadata']['source'] for chunk in gmail_chunks)
    print(f"Sources: {sources}")
    
    # Check content length distribution
    lengths = [len(chunk['text']) for chunk in gmail_chunks]
    avg_length = sum(lengths) / len(lengths)
    print(f"Average chunk length: {avg_length:.0f} characters")
    print(f"Length range: {min(lengths)} - {max(lengths)} characters")
    
    # Sample chunks from different emails
    print(f"\n📝 Sample Chunks (using your DocumentProcessor format):")
    
    unique_emails = {}
    for chunk in gmail_chunks:
        doc_id = chunk['metadata']['doc_id']  # This is the email ID
        if doc_id not in unique_emails:
            unique_emails[doc_id] = chunk
            
    for i, (doc_id, chunk) in enumerate(list(unique_emails.items())[:3]):
        print(f"\n--- Chunk {i+1} (Email: {doc_id[:8]}...) ---")
        print(f"Title: {chunk['metadata']['title']}")
        print(f"Source: {chunk['metadata']['source']}")
        print(f"Chunk {chunk['metadata']['chunk_index']+1}/{chunk['metadata']['total_chunks']}")
        print(f"Text: {chunk['text'][:200]}...")
    
    return gmail_chunks

async def test_search_simulation():
    """Simulate how these chunks would work in search using your existing format"""
    print(f"\n=== Testing Search Simulation ===")
    
    processor = MultiSourceProcessor()
    gmail_chunks = await processor._process_gmail()
    
    if not gmail_chunks:
        print("No chunks to test search with")
        return
    
    # Simulate search queries
    test_queries = [
        "job alert",
        "frontend developer", 
        "login notification",
        "newsletter",
        "team building"
    ]
    
    print(f"🔍 Testing search simulation with {len(test_queries)} queries:")
    
    for query in test_queries:
        matching_chunks = []
        query_lower = query.lower()
        
        for chunk in gmail_chunks:
            text_lower = chunk['text'].lower()  # Using 'text' field from your format
            if query_lower in text_lower:
                matching_chunks.append(chunk)
        
        print(f"\nQuery: '{query}' → {len(matching_chunks)} matches")
        
        if matching_chunks:
            best_match = matching_chunks[0]
            print(f"  Best match: {best_match['metadata']['title'][:50]}...")
            print(f"  Source: {best_match['metadata']['source']}")
            print(f"  Chunk: {best_match['metadata']['chunk_index']+1}/{best_match['metadata']['total_chunks']}")

async def test_incremental_updates():
    """Test incremental update functionality"""
    print(f"\n=== Testing Incremental Updates ===")
    
    processor = MultiSourceProcessor()
    
    # Test different time windows
    time_windows = [1, 6, 24]  # hours
    
    for hours in time_windows:
        recent_chunks = await processor.process_incremental_gmail(hours_back=hours)
        print(f"Last {hours} hours: {len(recent_chunks)} new chunks")

async def main():
    """Run all tests"""
    print("🚀 Testing Multi-Source Document Processing\n")
    
    try:
        # Test 1: Gmail chunk processing
        gmail_chunks = await test_gmail_chunks()
        
        # Test 2: Search simulation
        await test_search_simulation()
        
        # Test 3: Incremental updates
        await test_incremental_updates()
        
        print(f"\n✅ All tests completed successfully!")
        print(f"📈 Ready to integrate with vector database")
        
        return gmail_chunks
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    chunks = asyncio.run(main())