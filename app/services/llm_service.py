import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LLMService:
    """Service for generating suggestions using Together AI and Mistral"""
    
    def __init__(self):
        self.api_key = os.getenv("TOGETHER_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        self.session = None
        
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY environment variable is required")

    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _build_prompt(self, user_text: str, context: str, retrieved_chunks: List[str], task: str) -> str:
        """Build the prompt for the LLM based on the task type"""
        
        # Combine retrieved chunks into context
        relevant_content = "\n".join([f"- {chunk}" for chunk in retrieved_chunks[:5]])
        
        if task == "continue":
            prompt = f"""You are a writing assistant that helps users continue their writing while preserving their unique voice and style.

Based on the user's previous writing samples below, continue the given text in their authentic style:

PREVIOUS WRITING SAMPLES:
{relevant_content}

CONTEXT: {context}
CURRENT TEXT: {user_text}

Continue this text naturally in the user's writing style. Provide a brief, natural continuation (1-2 sentences max) that:
1. Maintains the user's tone and voice
2. Flows naturally from the current text
3. Stays relevant to the context
4. Sounds like the user wrote it

CONTINUATION:"""

        elif task == "complete":
            prompt = f"""You are a writing assistant that helps users complete their thoughts while preserving their unique voice and style.

Based on the user's previous writing samples below, complete the given text in their authentic style:

PREVIOUS WRITING SAMPLES:
{relevant_content}

CONTEXT: {context}
CURRENT TEXT: {user_text}

Complete this thought in the user's writing style. Provide a natural completion that:
1. Finishes the idea or sentence logically
2. Maintains the user's tone and voice
3. Stays relevant to the context
4. Sounds like the user wrote it

COMPLETION:"""

        elif task == "rephrase":
            prompt = f"""You are a writing assistant that helps users rephrase their writing while preserving their unique voice and style.

Based on the user's previous writing samples below, rephrase the given text in their authentic style:

PREVIOUS WRITING SAMPLES:
{relevant_content}

CONTEXT: {context}
CURRENT TEXT: {user_text}

Rephrase this text in the user's writing style. Provide an alternative version that:
1. Maintains the same meaning
2. Uses the user's typical vocabulary and tone
3. Stays relevant to the context
4. Sounds like the user wrote it

REPHRASE:"""

        else:
            # Default to continue
            return self._build_prompt(user_text, context, retrieved_chunks, "continue")
        
        return prompt
    
    async def generate_suggestions(
        self, 
        user_text: str, 
        context: str, 
        retrieved_chunks: List[str], 
        task: str = "continue",
        num_suggestions: int = 3,
        max_tokens: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate suggestions using the LLM"""
        
        if not retrieved_chunks:
            logger.warning("No retrieved chunks provided, generating fallback suggestions")
            return await self._generate_fallback_suggestions(user_text, task, num_suggestions)
        
        suggestions = []
        session = await self._get_session()
        
        # Generate multiple suggestions by making separate API calls
        # This gives us more variety than asking for multiple in one call
        for i in range(num_suggestions):
            try:
                prompt = self._build_prompt(user_text, context, retrieved_chunks, task)
                
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.7 + (i * 0.1),  # Vary temperature for diversity
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "stop": ["\n\n", "CONTEXT:", "CURRENT TEXT:", "PREVIOUS WRITING"]
                }
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                start_time = time.time()
                async with session.post(self.base_url, json=payload, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        generation_time = (time.time() - start_time) * 1000
                        
                        suggestion_text = result["choices"][0]["message"]["content"].strip()
                        
                        # Clean up the suggestion
                        suggestion_text = self._clean_suggestion(suggestion_text)
                        
                        if suggestion_text:
                            suggestions.append({
                                "text": suggestion_text,
                                "score": 0.9 - (i * 0.1),  # Decrease score for later suggestions
                                "reasoning": f"Generated using {self.model_name} based on similar content",
                                "generation_time_ms": int(generation_time)
                            })
                    else:
                        error_text = await response.text()
                        logger.error(f"LLM API error {response.status}: {error_text}")
                        
            except Exception as e:
                logger.error(f"Error generating suggestion {i+1}: {str(e)}")
                continue
                
            # Small delay between requests to avoid rate limiting
            if i < num_suggestions - 1:
                await asyncio.sleep(0.1)
        
        # If we didn't get any suggestions, provide fallbacks
        if not suggestions:
            logger.warning("No LLM suggestions generated, using fallbacks")
            return await self._generate_fallback_suggestions(user_text, task, num_suggestions)
        
        return suggestions
    
    def _clean_suggestion(self, text: str) -> str:
        """Clean up the generated suggestion text"""
        # Remove common artifacts from LLM generation
        text = text.strip()
        
        # Remove any quoted text or prefixes
        prefixes_to_remove = [
            "CONTINUATION:", "COMPLETION:", "REPHRASE:",
            "Here's a continuation:", "Here's a completion:", "Here's a rephrase:",
            "Continuation:", "Completion:", "Rephrase:"
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove quotes if the entire text is quoted
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        
        # Ensure reasonable length (trim if too long)
        if len(text) > 200:
            # Try to end at a sentence boundary
            sentences = text.split('.')
            if len(sentences) > 1:
                text = sentences[0] + '.'
            else:
                text = text[:200].rsplit(' ', 1)[0] + '...'
        
        return text
    
    async def _generate_fallback_suggestions(
        self, 
        user_text: str, 
        task: str, 
        num_suggestions: int
    ) -> List[Dict[str, Any]]:
        """Generate fallback suggestions when LLM fails"""
        
        fallback_suggestions = {
            "continue": [
                "continues with clear purpose and direction",
                "develops this idea further with specific examples",
                "builds upon this foundation naturally"
            ],
            "complete": [
                "requires careful consideration and planning",
                "represents an important step forward",
                "deserves our full attention and effort"
            ],
            "rephrase": [
                user_text.replace(" and ", " & "),
                user_text.replace(" but ", " however, "),
                user_text.replace(" because ", " since ")
            ]
        }
        
        suggestions = []
        fallbacks = fallback_suggestions.get(task, fallback_suggestions["continue"])
        
        for i, fallback in enumerate(fallbacks[:num_suggestions]):
            suggestions.append({
                "text": fallback,
                "score": 0.2,
                "reasoning": "Fallback suggestion - LLM generation failed",
                "generation_time_ms": 0
            })
        
        return suggestions
    
    def __del__(self):
        """Cleanup on destruction"""
        if self.session and not self.session.closed:
            # Note: This won't work in async context, but it's a safety net
            pass