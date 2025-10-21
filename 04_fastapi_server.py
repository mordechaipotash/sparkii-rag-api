#!/usr/bin/env python3
"""
SPARKII RAG: FastAPI REST API Server
=====================================

Purpose: Production REST API for the RAG system

Endpoints:
- POST /search - Semantic search with filters
- POST /ask - RAG question answering
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
from contextlib import asynccontextmanager
import uvicorn
from openai import OpenAI
import os

# Import from the local rag_retriever module using relative path
import importlib.util
import sys
from pathlib import Path

# Get the directory of this file for relative imports
CURRENT_DIR = Path(__file__).parent.resolve()
RAG_RETRIEVER_PATH = CURRENT_DIR / "03_rag_retriever_stella.py"

# Dynamically import the rag_retriever module
spec = importlib.util.spec_from_file_location("rag_retriever", str(RAG_RETRIEVER_PATH))
rag_retriever = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_retriever)

# Import the classes we need
SparkiiRetriever = rag_retriever.SparkiiRetriever
RetrievalFilters = rag_retriever.RetrievalFilters
QueryType = rag_retriever.QueryType

# ============================================================================
# FASTAPI APP WITH LIFESPAN
# ============================================================================

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    global retriever

    # Startup
    print("🚀 Starting Sparkii RAG API...")
    print("⏳ Retriever will initialize on first search request")
    # Don't load stella model on startup - it takes 2-3 minutes
    # Load it lazily on first /search or /ask request
    retriever = None
    print("✅ API ready! (model loads on first request)")

    yield  # Application runs here

    # Shutdown
    print("🔄 Shutting down Sparkii RAG API...")
    if retriever:
        print("📊 Closing connection pool...")
        retriever.close()
        print("✅ Connection pool closed")
    print("👋 Sparkii RAG API stopped")

app = FastAPI(
    title="Sparkii RAG API",
    description="Semantic search over 255K AI conversation messages",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - allow all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

class SearchResult(BaseModel):
    """Single search result"""
    id: str
    conversation_id: str
    message_index: int
    role: str
    content: str
    distance: float
    conversation_title: Optional[str] = None
    conversation_date: Optional[datetime] = None
    contains_code: Optional[bool] = None
    code_language: Optional[str] = None
    tools_used: Optional[List[str]] = None
    previous_messages: Optional[List[Dict[str, Any]]] = None

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
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API info"""
    return {
        "name": "Sparkii RAG API",
        "version": "1.0.0",
        "description": "Semantic search over 255K AI conversation messages",
        "endpoints": {
            "search": "/search - Semantic search with filters",
            "ask": "/ask - RAG question answering",
            "stats": "/stats - System statistics",
            "health": "/health - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "retriever_loaded": retriever is not None
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Semantic search endpoint

    Example:
    ```
    POST /search
    {
        "query": "How did I fix the authentication bug?",
        "top_k": 10,
        "filters": {
            "contains_code": true,
            "code_language": "python"
        }
    }
    ```
    """
    global retriever

    # Lazy initialization: load stella model on first request
    if not retriever:
        print("⏳ First search request - loading stella model (this may take 2-3 minutes)...")
        try:
            retriever = SparkiiRetriever()
            print("✅ Retriever initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize retriever: {e}")
            raise HTTPException(status_code=503, detail=f"Failed to initialize retriever: {str(e)}")

    try:
        # Build filters
        filters = None
        if request.filters:
            filters = RetrievalFilters(
                contains_code=request.filters.get('contains_code'),
                code_language=request.filters.get('code_language'),
                user_intent=request.filters.get('user_intent'),
                technical_depth=request.filters.get('technical_depth'),
                response_type=request.filters.get('response_type'),
                tools_used=request.filters.get('tools_used'),
                mcp_tools_used=request.filters.get('mcp_tools_used'),
                role=request.filters.get('role'),
                frustration_indicator=request.filters.get('frustration_indicator'),
                urgency_level=request.filters.get('urgency_level')
            )

        # Execute search
        results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            filters=filters,
            include_context=request.include_context,
            distance_threshold=request.distance_threshold
        )

        # Determine query type
        query_type = retriever.route_query(request.query).value

        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
            query_type=query_type
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/smart", response_model=SearchResponse)
async def smart_search(request: SearchRequest):
    """
    Smart search with automatic query routing

    This endpoint automatically determines the best retrieval strategy
    based on query analysis (debugging, code, conceptual, etc.)

    Example:
    ```
    POST /search/smart
    {
        "query": "How did I fix the authentication bug?",
        "top_k": 10
    }
    ```
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        # Use smart routing
        results = retriever.search_with_routing(
            query=request.query,
            top_k=request.top_k
        )

        query_type = retriever.route_query(request.query).value

        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
            query_type=query_type
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    RAG question answering endpoint with Claude Haiku 4.5

    This endpoint:
    1. Retrieves relevant messages using smart routing
    2. Formats them as context
    3. Generates a synthesized answer using Claude Haiku 4.5

    Example:
    ```
    POST /ask
    {
        "question": "How do I deploy to production?",
        "top_k": 5
    }
    ```
    """
    global retriever

    # Lazy initialization: load stella model on first request
    if not retriever:
        print("⏳ First ask request - loading stella model (this may take 2-3 minutes)...")
        try:
            retriever = SparkiiRetriever()
            print("✅ Retriever initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize retriever: {e}")
            raise HTTPException(status_code=503, detail=f"Failed to initialize retriever: {str(e)}")

    try:
        # Build filters
        filters = None
        if request.filters:
            filters = RetrievalFilters(
                contains_code=request.filters.get('contains_code'),
                code_language=request.filters.get('code_language'),
                user_intent=request.filters.get('user_intent'),
                technical_depth=request.filters.get('technical_depth'),
                response_type=request.filters.get('response_type'),
                tools_used=request.filters.get('tools_used'),
                mcp_tools_used=request.filters.get('mcp_tools_used'),
                role=request.filters.get('role'),
                frustration_indicator=request.filters.get('frustration_indicator'),
                urgency_level=request.filters.get('urgency_level')
            )

        # Use smart routing for better retrieval
        results = retriever.search_with_routing(
            query=request.question,
            top_k=request.top_k
        )

        # If no results found, return early
        if not results:
            return AskResponse(
                question=request.question,
                answer="I couldn't find any relevant information in your conversation history to answer this question.",
                sources=[],
                confidence=0.0
            )

        # Format context for Claude
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[Source {i}] Role: {r['role']}")
            if r.get('conversation_title'):
                context_parts.append(f"From conversation: {r['conversation_title']}")
            if r.get('contains_code') and r.get('code_language'):
                context_parts.append(f"Contains {r['code_language']} code")
            context_parts.append(f"Content: {r['content']}")
            context_parts.append("")  # blank line

        context = "\n".join(context_parts)

        # Build prompt for Claude Haiku 4.5
        system_prompt = """You are a helpful assistant that answers questions based on the user's past conversation history.

Your task is to:
1. Analyze the provided conversation excerpts
2. Synthesize a clear, concise answer to the user's question
3. Reference specific sources when making claims
4. If the sources don't fully answer the question, say so
5. Be direct and helpful - this is the user's own conversation history

Format your answer naturally, and cite sources like [Source 1] when referencing them."""

        user_prompt = f"""Question: {request.question}

Relevant conversation excerpts:

{context}

Please provide a clear, synthesized answer based on these sources."""

        # Call Claude Haiku 4.5 via OpenRouter
        response = openrouter_client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )

        answer = response.choices[0].message.content

        # Calculate confidence score based on retrieval distances
        # Lower distance = higher confidence (distance is 0-2, we invert it)
        avg_distance = sum(r['distance'] for r in results) / len(results)
        confidence = max(0.0, min(1.0, 1.0 - (avg_distance / 2.0)))

        return AskResponse(
            question=request.question,
            answer=answer,
            sources=results,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get system statistics

    Returns counts and metrics about the RAG system
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        cursor = retriever.conn.cursor(cursor_factory=RealDictCursor)

        # Get statistics
        cursor.execute("""
            SELECT
                COUNT(*) as total_messages,
                COUNT(DISTINCT conversation_id) as total_conversations,
                SUM(CASE WHEN contains_code THEN 1 ELSE 0 END) as messages_with_code,
                AVG(classification_confidence) as average_confidence
            FROM message_embeddings
        """)

        stats = cursor.fetchone()
        cursor.close()

        return StatsResponse(
            total_messages=stats['total_messages'],
            total_conversations=stats['total_conversations'],
            messages_with_code=stats['messages_with_code'],
            average_confidence=float(stats['average_confidence'] or 0.0)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SPARKII RAG: FastAPI Server")
    print("="*80 + "\n")
    print("📡 Starting server on http://localhost:8000")
    print("📖 API docs at http://localhost:8000/docs")
    print("🔄 Interactive docs at http://localhost:8000/redoc")
    print("\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
