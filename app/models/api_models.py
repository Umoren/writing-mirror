from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


class SuggestRequest(BaseModel):
    """Request model for the /api/suggest endpoint"""
    text: str = Field(..., description="Current text being written")
    context: Optional[str] = Field(None, description="Additional context or writing purpose")
    task: str = Field(default="continue", description="Type of suggestion: 'continue', 'complete', 'rephrase'")
    num_suggestions: int = Field(default=3, ge=1, le=10, description="Number of suggestions to return")
    max_length: int = Field(default=100, ge=10, le=500, description="Maximum length of each suggestion")


class Source(BaseModel):
    """Source document information for a suggestion"""
    doc_id: str = Field(..., description="Document ID from Notion")
    title: str = Field(..., description="Document title")
    similarity: float = Field(..., ge=0, le=1, description="Similarity score")
    chunk_text: Optional[str] = Field(None, description="Relevant chunk text")


class Suggestion(BaseModel):
    """Individual suggestion with metadata"""
    text: str = Field(..., description="Suggested text continuation")
    score: float = Field(..., ge=0, le=1, description="Confidence score")
    reasoning: Optional[str] = Field(None, description="Why this suggestion was generated")


class PerformanceStats(BaseModel):
    """Performance metrics for the suggestion request"""
    total_time_ms: int = Field(..., description="Total processing time in milliseconds")
    embedding_time_ms: int = Field(..., description="Time to generate query embedding")
    search_time_ms: int = Field(..., description="Time to search vector database")
    generation_time_ms: int = Field(..., description="Time to generate suggestions")
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    chunks_processed: int = Field(..., description="Number of chunks processed")


class SuggestResponse(BaseModel):
    """Response model for the /api/suggest endpoint"""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request identifier")
    suggestions: List[Suggestion] = Field(..., description="Generated suggestions")
    sources: List[Source] = Field(..., description="Source documents used for suggestions")
    stats: PerformanceStats = Field(..., description="Performance metrics")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, Any] = Field(..., description="Status of dependent services")
    version: str = Field(default="1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    trace_id: str = Field(..., description="Request trace ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")