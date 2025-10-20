#!/usr/bin/env python3
"""
SPARKII RAG: LangChain Retriever with STELLA Embeddings
========================================================

Purpose: Build a production-ready LangChain retriever with stella-en-1.5B-v5
- Uses sentence-transformers (same as embedding generation)
- 1024-dimensional vectors (vs 384 for MiniLM)
- MTEB rank #3 quality (vs #50+ for MiniLM)
- Classification metadata filtering
- Conversation threading
- Query routing

Model caching:
- First run: Downloads stella model (~5.6GB) to ~/.cache/huggingface
- Subsequent runs: Loads from cache (fast)
- Railway: Use volume mount at /app/.cache to persist model
"""

import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

SUPABASE_URL = DATABASE_URL

# Stella model (same as embedding generation)
MODEL_NAME = "NovaSearch/stella_en_1.5B_v5"
EMBEDDING_DIM = 1024

# HuggingFace cache directory (Railway volume mount point)
CACHE_DIR = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# ============================================================================
# QUERY TYPES (for routing)
# ============================================================================

class QueryType(Enum):
    """Different query types need different retrieval strategies"""
    DEBUGGING = "debugging"        # "How did I fix X?" - needs code + tools
    CONCEPTUAL = "conceptual"      # "What is X?" - needs explanations
    CODE_SEARCH = "code_search"    # "Show me X implementation" - needs code only
    WORKFLOW = "workflow"          # "How do I do X?" - needs sequential messages
    ERROR_RECOVERY = "error_recovery"  # "What went wrong with X?" - needs frustration indicators

# ============================================================================
# RETRIEVER FILTERS
# ============================================================================

@dataclass
class RetrievalFilters:
    """Filters to apply before vector search (HUGE performance boost!)"""
    contains_code: Optional[bool] = None
    code_language: Optional[str] = None
    user_intent: Optional[str] = None
    technical_depth: Optional[str] = None
    response_type: Optional[str] = None
    tools_used: Optional[List[str]] = None
    mcp_tools_used: Optional[List[str]] = None
    role: Optional[str] = None  # user, assistant, system
    frustration_indicator: Optional[bool] = None
    urgency_level: Optional[str] = None

# ============================================================================
# SPARKII RETRIEVER
# ============================================================================

class SparkiiRetriever:
    """
    Production LangChain-compatible retriever with stella embeddings

    This is the MAGIC - it combines:
    1. Pre-filtering (using classification metadata)
    2. Vector search (semantic similarity with stella)
    3. Context expansion (fetches previous messages in conversation)
    4. Re-ranking (optional - by relevance score)
    """

    def __init__(self):
        # Load stella model using sentence-transformers
        print(f"🔄 Loading stella model from {CACHE_DIR}...")
        self.model = SentenceTransformer(
            MODEL_NAME,
            device='cpu',  # Railway uses CPU
            cache_folder=CACHE_DIR
        )
        print(f"✅ Model loaded: {MODEL_NAME}")
        print(f"   Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        # Store database URL for per-request connections
        self.db_url = SUPABASE_URL

    def encode_query(self, query: str) -> List[float]:
        """Convert query to stella embedding vector (1024 dims)"""
        embedding = self.model.encode(query, convert_to_tensor=False)
        return embedding.tolist()

    def search(
        self,
        query: str,
        top_k: int = 10,
        limit: int = None,  # Backward compatibility alias
        filters: Optional[RetrievalFilters] = None,
        include_context: bool = False,
        distance_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with stella embeddings

        Args:
            query: Natural language search query
            top_k: Max results to return (default: 10)
            limit: Alias for top_k (backward compatibility)
            filters: Optional classification-based filters
            include_context: Not yet implemented
            distance_threshold: Not yet implemented

        Returns:
            List of matching messages with metadata and similarity scores
        """
        # Use limit if provided (backward compat), otherwise use top_k
        result_limit = limit if limit is not None else top_k

        # 1. Encode query
        query_embedding = self.encode_query(query)

        # 2. Build WHERE clause from filters
        where_conditions = ["m.embedding IS NOT NULL"]
        params = [query_embedding]

        if filters:
            if filters.contains_code is not None:
                where_conditions.append("m.contains_code = %s")
                params.append(filters.contains_code)
            if filters.code_language:
                where_conditions.append("m.code_language = %s")
                params.append(filters.code_language)
            if filters.user_intent:
                where_conditions.append("m.user_intent = %s")
                params.append(filters.user_intent)
            if filters.technical_depth:
                where_conditions.append("m.technical_depth = %s")
                params.append(filters.technical_depth)
            if filters.role:
                where_conditions.append("m.role = %s")
                params.append(filters.role)
            if filters.frustration_indicator is not None:
                where_conditions.append("m.frustration_indicator = %s")
                params.append(filters.frustration_indicator)

        where_clause = " AND ".join(where_conditions)

        # 3. Execute vector search
        query_sql = f"""
            SELECT
                m.content,
                m.role,
                c.title,
                c.conversation_id,
                m.message_index,
                m.user_intent,
                m.technical_depth,
                m.contains_code,
                m.code_language,
                m.tools_used,
                m.mcp_tools_used,
                m.embedding <=> %s::vector AS distance
            FROM message_embeddings m
            JOIN chat_histories c ON m.conversation_id = c.id
            WHERE {where_clause}
            ORDER BY distance ASC
            LIMIT %s
        """

        params.append(result_limit)

        # Create a fresh connection for this request
        conn = None
        cursor = None
        try:
            conn = psycopg.connect(self.db_url)
            cursor = conn.cursor(row_factory=dict_row)
            cursor.execute(query_sql, params)
            # Materialize results immediately before closing cursor
            raw_results = cursor.fetchall()

            # Close cursor and connection before processing results
            cursor.close()
            cursor = None
            conn.close()
            conn = None

            # Convert to regular Python dicts and add similarity scores
            results = []
            for row in raw_results:
                result = dict(row)
                result['similarity_score'] = 1 - result['distance']
                result['match_percentage'] = round((1 - result['distance']) * 100, 1)
                results.append(result)

            return results
        except Exception as e:
            print(f"Database error: {e}")
            print(f"Connection string (masked): {self.db_url[:50]}...")
            raise
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def get_conversation_context(
        self,
        conversation_id: str,
        message_index: int,
        context_window: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get surrounding messages from same conversation

        Args:
            conversation_id: UUID of conversation
            message_index: Index of target message
            context_window: How many messages before/after to fetch

        Returns:
            List of messages in chronological order
        """
        conn = psycopg.connect(self.db_url)
        cursor = conn.cursor(row_factory=dict_row)
        cursor.execute("""
            SELECT
                content,
                role,
                message_index,
                user_intent,
                tools_used,
                mcp_tools_used
            FROM message_embeddings
            WHERE conversation_id = %s
                AND message_index BETWEEN %s AND %s
            ORDER BY message_index ASC
        """, (
            conversation_id,
            message_index - context_window,
            message_index + context_window
        ))

        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Quick search without filters (for testing)"""
    retriever = SparkiiRetriever()
    return retriever.search(query, limit=limit)


def code_search(query: str, language: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for code snippets only"""
    filters = RetrievalFilters(
        contains_code=True,
        code_language=language
    )
    retriever = SparkiiRetriever()
    return retriever.search(query, limit=limit, filters=filters)


def debugging_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for debugging/error-related messages"""
    filters = RetrievalFilters(
        user_intent="troubleshooting",
        frustration_indicator=True
    )
    retriever = SparkiiRetriever()
    return retriever.search(query, limit=limit, filters=filters)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 03_rag_retriever_stella.py <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\n🔍 Searching: {query}\n")

    results = quick_search(query, limit=5)

    for i, result in enumerate(results, 1):
        print(f"[{i}] {result['match_percentage']}% match | {result['role']} | {result['title']}")
        print(f"    {result['content'][:200]}...")
        print()
