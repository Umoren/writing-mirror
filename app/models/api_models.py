"""
API models for the Voice Writing Assistant
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class SuggestRequest(BaseModel):
    """Request model for the suggest endpoint"""
    text: str
    context: Optional[str] = None
    task: str = "continue"
    num_suggestions: int = 3

class Suggestion(BaseModel):
    """Model for a suggestion"""
    text: str
    score: float

class SuggestResponse(BaseModel):
    """Response model for the suggest endpoint"""
    trace_id: str
    suggestions: List[Suggestion]
    sources: List[Dict[str, Any]]
    stats: Dict[str, Any]