"""
Notion API integration service
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from notion_client import AsyncClient

logger = logging.getLogger(__name__)

class NotionService:
    """Service for interacting with the Notion API"""
    
    def __init__(self, api_key: str, database_id: str):
        """
        Initialize the Notion service
        
        Args:
            api_key: Notion API key
            database_id: ID of the Notion database to use
        """
        self.client = AsyncClient(auth=api_key)
        self.database_id = database_id
        logger.info(f"Notion service initialized for database: {database_id}")
    
    async def list_database_pages(self, 
                          filter_condition: Optional[Dict[str, Any]] = None, 
                          page_size: int = 100) -> List[Dict[str, Any]]:
        """
        List pages in the Notion database
        
        Args:
            filter_condition: Optional filter condition for the query
            page_size: Number of pages to return per request
            
        Returns:
            List[Dict[str, Any]]: List of pages
        """
        try:
            # Query the database
            pages = []
            cursor = None
            
            while True:
                # Build query parameters
                query_params = {
                    "database_id": self.database_id,
                    "page_size": page_size
                }
                
                if filter_condition:
                    query_params["filter"] = filter_condition
                
                if cursor:
                    query_params["start_cursor"] = cursor
                
                # Execute query
                response = await self.client.databases.query(**query_params)
                
                # Extract pages
                pages.extend(response["results"])
                
                # Check if there are more pages
                if not response.get("has_more", False):
                    break
                
                # Get next cursor
                cursor = response.get("next_cursor")
                
                # Respect Notion API rate limits (3 requests per second)
                await asyncio.sleep(0.34)
            
            logger.info(f"Retrieved {len(pages)} pages from database {self.database_id}")
            return pages
        
        except Exception as e:
            logger.error(f"Error listing database pages: {e}")
            raise
    
    async def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """
        Get content of a Notion page
        
        Args:
            page_id: ID of the page
            
        Returns:
            Dict[str, Any]: Page content
        """
        try:
            # Get page metadata
            page = await self.client.pages.retrieve(page_id=page_id)
            
            # Get page blocks
            blocks = []
            cursor = None
            
            while True:
                # Build query parameters
                query_params = {
                    "block_id": page_id,
                    "page_size": 100
                }
                
                if cursor:
                    query_params["start_cursor"] = cursor
                
                # Execute query
                response = await self.client.blocks.children.list(**query_params)
                
                # Extract blocks
                blocks.extend(response["results"])
                
                # Check if there are more blocks
                if not response.get("has_more", False):
                    break
                
                # Get next cursor
                cursor = response.get("next_cursor")
                
                # Respect Notion API rate limits
                await asyncio.sleep(0.34)
            
            # Extract page properties
            title = self._extract_page_title(page)
            created_time = page.get("created_time")
            last_edited_time = page.get("last_edited_time")
            
            # Extract page content
            content = {
                "id": page_id,
                "title": title,
                "created_time": created_time,
                "last_edited_time": last_edited_time,
                "blocks": blocks
            }
            
            logger.info(f"Retrieved content for page {page_id}")
            return content
        
        except Exception as e:
            logger.error(f"Error getting page content: {e}")
            raise
    
    async def extract_text_from_page(self, page: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text content from a Notion page
        
        Args:
            page: Page object
            
        Returns:
            Dict[str, Any]: Page content with extracted text
        """
        try:
            # Extract title and metadata
            title = self._extract_page_title(page)
            created_time = page.get("created_time")
            last_edited_time = page.get("last_edited_time")
            
            # Extract content from properties (for database entries)
            content = ""
            properties = page.get("properties", {})
            
            # Look for a Content property
            for prop_name, prop_value in properties.items():
                if prop_name == "Content":  # Use exact property name
                    # Rich text property
                    if prop_value.get("type") == "rich_text":
                        rich_texts = prop_value.get("rich_text", [])
                        content = self._extract_rich_text(rich_texts)
                    break

            # Build result
            result = {
                "id": page["id"],
                "title": title,
                "created_time": created_time,
                "last_edited_time": last_edited_time,
                "content": content
            }
            
            logger.info(f"Extracted text from page {page['id']}")
            return result
        
        except Exception as e:
            logger.error(f"Error extracting text from page: {e}")
            raise
    
    async def process_all_documents(self, 
                            last_sync_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process all documents in the database
        
        Args:
            last_sync_time: Optional timestamp for incremental sync
            
        Returns:
            List[Dict[str, Any]]: List of processed documents
        """
        try:
            # Build filter for incremental sync
            filter_condition = None
            if last_sync_time:
                filter_condition = {
                    "timestamp": "last_edited_time",
                    "last_edited_time": {
                        "after": last_sync_time
                    }
                }
            
            # Get pages
            pages = await self.list_database_pages(filter_condition=filter_condition)
            
            # Process pages
            processed_docs = []
            for page in pages:
                # Extract text content - pass the entire page object, not just the ID
                doc = await self.extract_text_from_page(page)
                
                # Add to processed documents
                processed_docs.append(doc)
                
                # Rate limit to avoid API limits (3 requests per second)
                await asyncio.sleep(0.34)
            
            logger.info(f"Processed {len(processed_docs)} documents")
            return processed_docs
        
        except Exception as e:
            logger.error(f"Error processing all documents: {e}")
            raise
    
    def _extract_page_title(self, page: Dict[str, Any]) -> str:
        """
        Extract title from a page
        
        Args:
            page: Page object
            
        Returns:
            str: Page title
        """
        try:
            # Extract title from properties
            properties = page.get("properties", {})
            
            # Look for title property
            for prop_name, prop_value in properties.items():
                if prop_name == "Title" or prop_value.get("type") == "title":
                    title_items = prop_value.get("title", [])
                    if title_items:
                        return "".join([item.get("plain_text", "") for item in title_items])
            
            # Fallback to page ID if title not found
            return f"Untitled Page ({page['id']})"
        
        except Exception as e:
            logger.error(f"Error extracting page title: {e}")
            return f"Untitled Page ({page.get('id', 'unknown')})"
    
    def _extract_rich_text(self, rich_text: List[Dict[str, Any]]) -> str:
        """
        Extract text from rich text array

        Args:
            rich_text: Rich text array

        Returns:
            str: Extracted text
        """
        if not rich_text:
            return ""

        return "".join([item.get("plain_text", "") for item in rich_text])

    def _extract_text_from_block(self, block: Dict[str, Any]) -> Optional[str]:
        """
        Extract text from a block
        
        Args:
            block: Block object
            
        Returns:
            Optional[str]: Extracted text or None if no text
        """
        try:
            block_type = block.get("type")
            
            # Handle different block types
            if block_type == "paragraph":
                return self._extract_rich_text(block.get("paragraph", {}).get("rich_text", []))
            elif block_type == "heading_1":
                return self._extract_rich_text(block.get("heading_1", {}).get("rich_text", []))
            elif block_type == "heading_2":
                return self._extract_rich_text(block.get("heading_2", {}).get("rich_text", []))
            elif block_type == "heading_3":
                return self._extract_rich_text(block.get("heading_3", {}).get("rich_text", []))
            elif block_type == "bulleted_list_item":
                return "• " + self._extract_rich_text(block.get("bulleted_list_item", {}).get("rich_text", []))
            elif block_type == "numbered_list_item":
                return "1. " + self._extract_rich_text(block.get("numbered_list_item", {}).get("rich_text", []))
            elif block_type == "to_do":
                checked = block.get("to_do", {}).get("checked", False)
                prefix = "[x] " if checked else "[ ] "
                return prefix + self._extract_rich_text(block.get("to_do", {}).get("rich_text", []))
            elif block_type == "toggle":
                return self._extract_rich_text(block.get("toggle", {}).get("rich_text", []))
            elif block_type == "code":
                code_text = self._extract_rich_text(block.get("code", {}).get("rich_text", []))
                language = block.get("code", {}).get("language", "")
                return f"```{language}\n{code_text}\n```"
            elif block_type == "quote":
                return "> " + self._extract_rich_text(block.get("quote", {}).get("rich_text", []))
            elif block_type == "callout":
                return self._extract_rich_text(block.get("callout", {}).get("rich_text", []))
            else:
                return None
        
        except Exception as e:
            logger.error(f"Error extracting text from block: {e}")
            return None