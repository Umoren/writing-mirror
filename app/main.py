"""
Voice Writing Assistant
Main FastAPI application entry point
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

# Import our new suggest router
from .api.suggest import router as suggest_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Voice Writing Assistant",
    description="A personalized AI tool that helps improve your writing while maintaining your authentic voice",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the suggest API router
app.include_router(suggest_router, prefix="/api", tags=["suggestions"])

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint"""
    return JSONResponse(
        content={
            "app": "Voice Writing Assistant",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "docs": "/docs",
            "endpoints": {
                "suggest": "/api/suggest",
                "health": "/health",
                "api_health": "/api/health",
                "status": "/api/status"
            }
        }
    )

# Health check endpoint (keeping your existing one for compatibility)
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

# API status endpoint - UPDATED to reflect current architecture
@app.get("/api/status", tags=["api"])
async def api_status():
    """API status endpoint with more details"""
    return {
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "suggest_api": "available",
            "vector_search": "available",
            "embedding_service": "available",
            "vector_database": "available"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Global exception on {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )