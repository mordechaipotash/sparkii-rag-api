#!/usr/bin/env python3
"""
SPARKII RAG: FastAPI REST API Server
=====================================

Purpose: Production REST API for the RAG system

Endpoints:
- POST /search - Semantic search with filters
- POST /ask - RAG question answering
- POST /retrieve - Direct retrieval with filters
- GET /stats - System statistics
- GET /health - Health check

This makes the RAG system accessible from ANYWHERE:
- Custom Claude commands
- Web apps
- Mobile apps
- Other services
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uvicorn
from openai import OpenAI
import os
import logging
import sys

# Clean import from app package
from app.retriever import SparkiiRetriever, RetrievalFilters, QueryType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log startup environment
logger.info(f"🐍 Python version: {sys.version}")
logger.info(f"📁 Working directory: {os.getcwd()}")
logger.info(f"📦 App module location: {__file__}")

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Sparkii RAG API",
    description="Semantic search over 255K AI conversation messages",
    version="1.0.0"
)

# CORS - allow all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global retriever instance
retriever: Optional[SparkiiRetriever] = None

# OpenRouter client for LLM answer generation
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SearchRequest(BaseModel):
    """Search request schema"""
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    include_context: bool = Field(default=True, description="Include previous messages")
    distance_threshold: float = Field(default=1.0, ge=0.0, le=2.0, description="Max cosine distance")

class AskRequest(BaseModel):
    """RAG question answering request"""
    question: str = Field(..., description="Question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context messages")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")

class RetrieveRequest(BaseModel):
    """Direct retrieval request"""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")

class SearchResponse(BaseModel):
    """Search response schema"""
    query: str
    results: List[Dict[str, Any]]
    total: int
    query_type: Optional[str] = None

class AskResponse(BaseModel):
    """RAG answer response"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: Optional[float] = None

class StatsResponse(BaseModel):
    """System statistics"""
    total_messages: int
    total_conversations: int
    messages_with_code: int
    average_confidence: float

# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize retriever on startup"""
    global retriever
    logger.info("=" * 60)
    logger.info("🚀 Starting Sparkii RAG API...")
    logger.info(f"📊 Environment: Railway Pro (8GB RAM, 8 vCPU)")
    logger.info(f"🔑 OpenRouter API Key: {'✅ Set' if OPENROUTER_API_KEY else '❌ Missing'}")
    logger.info(f"🗄️  Database URL: {'✅ Set' if os.getenv('DATABASE_URL') else '❌ Missing'}")
    logger.info("=" * 60)

    try:
        logger.info("📥 Initializing SparkiiRetriever...")
        retriever = SparkiiRetriever()
        logger.info("✅ Retriever initialized successfully!")
        logger.info(f"📊 Model loaded: sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"🎯 Ready to serve requests!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize retriever: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown():
    """Clean up on shutdown"""
    global retriever
    if retriever:
        logger.info("Shutting down retriever...")
        retriever = None

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Sparkii RAG API",
        "version": "1.0.0",
        "description": "Semantic search over 255K AI conversation messages",
        "endpoints": {
            "/health": "Health check",
            "/stats": "System statistics",
            "/search": "POST - Semantic search with filters",
            "/ask": "POST - RAG question answering",
            "/retrieve": "POST - Direct retrieval"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    return {
        "status": "healthy",
        "retriever": "ready",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        stats = retriever.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Semantic search with metadata filtering"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        # Convert filters dict to RetrievalFilters
        filters_obj = None
        if request.filters:
            filters_obj = RetrievalFilters(**request.filters)

        # Perform search
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            filters=filters_obj,
            include_context=request.include_context,
            distance_threshold=request.distance_threshold
        )

        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
            query_type="semantic_search"
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """RAG question answering"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        # Get relevant context
        filters_obj = None
        if request.filters:
            filters_obj = RetrievalFilters(**request.filters)

        context_results = retriever.search(
            query=request.question,
            top_k=request.top_k,
            filters=filters_obj,
            include_context=True,
            distance_threshold=0.8
        )

        # Build context for LLM
        context_text = "\n\n".join([
            f"[Message {i+1}] {r['role']}: {r['content'][:500]}"
            for i, r in enumerate(context_results)
        ])

        # Generate answer using OpenRouter
        response = openrouter_client.chat.completions.create(
            model="anthropic/claude-3.5-haiku",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant answering questions based on conversation history. Use the provided context to answer accurately."
                },
                {
                    "role": "user",
                    "content": f"Context from conversations:\n{context_text}\n\nQuestion: {request.question}\n\nAnswer based on the context:"
                }
            ],
            temperature=0.3,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        return AskResponse(
            question=request.question,
            answer=answer,
            sources=context_results,
            confidence=0.85 if len(context_results) >= 3 else 0.6
        )
    except Exception as e:
        logger.error(f"Ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """Direct retrieval endpoint for custom processing"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        filters_obj = None
        if request.filters:
            filters_obj = RetrievalFilters(**request.filters)

        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            filters=filters_obj,
            include_context=False
        )

        return {
            "query": request.query,
            "results": results,
            "total": len(results)
        }
    except Exception as e:
        logger.error(f"Retrieve error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
