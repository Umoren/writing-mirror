"""
Test script for the Notion service
"""
import sys
import os
import asyncio
import json
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the app package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

from app.services.notion_service import NotionService

async def test_notion_service():
    """Test the Notion service functionality"""
    # Get API key and database ID from environment
    api_key = os.getenv("NOTION_API_KEY")
    database_id = os.getenv("NOTION_DATABASE_ID")
    
    if not api_key or not database_id:
        print("ERROR: NOTION_API_KEY and NOTION_DATABASE_ID environment variables must be set.")
        return
    
    # Initialize the Notion service
    service = NotionService(api_key=api_key, database_id=database_id)
    
    # List database pages
    print(f"Listing pages in database {database_id}...")
    pages = await service.list_database_pages(page_size=5)  # Limit to 5 for testing
    
    print(f"Found {len(pages)} pages")
    
    if not pages:
        print("No pages found in the database.")
        return
    
    # Get the first page
    first_page = pages[0]
    page_id = first_page["id"]
    
    # Print debug info about page properties
    print("\nPage properties:")
    for prop_name, prop_value in first_page.get("properties", {}).items():
        print(f"- {prop_name}: {prop_value.get('type')}")

        # Show more details for the Content property
        if prop_name == "Content":
            if prop_value.get("type") == "rich_text":
                rich_texts = prop_value.get("rich_text", [])
                if rich_texts:
                    print(f"  Content preview: {rich_texts[0].get('plain_text', '')[:50]}...")
                else:
                    print("  Content is empty")
            else:
                print(f"  Content type: {prop_value.get('type')}")

    print(f"\nExtracting text from page {page_id}...")
    
    # Extract text from the page
    page_text = await service.extract_text_from_page(first_page)
    
    # Print page information
    print(f"Page title: {page_text['title']}")
    print(f"Created: {page_text['created_time']}")
    print(f"Last edited: {page_text['last_edited_time']}")
    
    # Check if content is available
    if 'content' in page_text and page_text['content']:
        content_length = len(page_text['content'])
        print(f"Content length: {content_length} characters")

        # Show a preview of the content
        preview_length = min(500, content_length)
        print(f"\nContent preview:\n{page_text['content'][:preview_length]}...")
    else:
        print("No content found in the page")

        # Try getting block content directly
        print("\nAttempting to get block content directly...")
        block_content = await service.get_page_content(page_id)
        print(f"Number of blocks: {len(block_content.get('blocks', []))}")

        if block_content.get('blocks'):
            first_block = block_content['blocks'][0]
            print(f"First block type: {first_block.get('type')}")
            print(f"First block content preview:")
            print(json.dumps(first_block, indent=2)[:300] + "...")
    
    # Save the extracted text to a JSON file
    with open("page_extract_example.json", "w") as f:
        json.dump(page_text, f, indent=2)
    
    print(f"\nFull extracted text saved to page_extract_example.json")

    # Test processing multiple documents
    print("\nProcessing all documents...")
    all_docs = await service.process_all_documents()
    print(f"Processed {len(all_docs)} documents")

    # Save all processed documents to a JSON file
    with open("all_docs_example.json", "w") as f:
        json.dump(all_docs[:5], f, indent=2)  # Save first 5 for brevity

    print(f"First 5 processed documents saved to all_docs_example.json")

if __name__ == "__main__":
    asyncio.run(test_notion_service())