"""
Configuration utilities for the Voice Writing Assistant
"""
import os
from pydantic import BaseSettings, Field
from dotenv import load_dotenv
from functools import lru_cache

# Load .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # API Keys
    together_api_key: str = Field(..., env="TOGETHER_API_KEY")
    notion_api_key: str = Field(..., env="NOTION_API_KEY")
    
    # Redis configuration
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    redis_password: str = Field(None, env="REDIS_PASSWORD")
    
    # Qdrant configuration
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    qdrant_collection_name: str = Field("voice_writing_assistant", env="QDRANT_COLLECTION_NAME")
    
    # Embedding model configuration
    embedding_model_name: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL_NAME")
    
    # LLM configuration
    model_name: str = Field("mistralai/Mistral-7B-Instruct-v0.3", env="MODEL_NAME")
    
    # Notion integration
    notion_database_id: str = Field(..., env="NOTION_DATABASE_ID")
    
    class Config:
        """Pydantic config"""
        case_sensitive = True
        env_file = ".env"


@lru_cache()
def load_config() -> Settings:
    """
    Load the application configuration with caching
    
    Returns:
        Settings: Application configuration
    """
    return Settings()