#!/usr/bin/env python3
"""
SPARKII RAG: LangChain Retriever with STELLA Embeddings
========================================================

Purpose: Build a production-ready LangChain retriever with stella-en-1.5B-v5
- Uses same model for query encoding as embedding generation
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

import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

SUPABASE_URL = DATABASE_URL

# Stella model (same as embedding generation)
MODEL_NAME = "dunzhang/stella_en_1.5B_v5"
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
        # Load stella model
        print(f"🔄 Loading stella model from {CACHE_DIR}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        self.model.eval()
        print(f"✅ Stella model loaded: {MODEL_NAME}")

        # Connect to database
        print("🔄 Connecting to Supabase...")
        self.conn = psycopg2.connect(SUPABASE_URL)
        print("✅ Connected to Supabase")

    def encode_query(self, query: str) -> List[float]:
        """Convert query to stella embedding vector (1024 dims)"""
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=False)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].tolist()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[RetrievalFilters] = None,
        include_context: bool = True,
        distance_threshold: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant messages with optional filters

        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Metadata filters to apply
            include_context: Include previous messages from conversation
            distance_threshold: Maximum cosine distance (0-2, lower = more similar)

        Returns:
            List of dicts with message content, metadata, and similarity scores
        """
        # Encode query
        query_embedding = self.encode_query(query)

        # Build SQL with filters
        where_clauses = []
        params = []

        if filters:
            if filters.contains_code is not None:
                where_clauses.append("contains_code = %s")
                params.append(filters.contains_code)

            if filters.code_language:
                where_clauses.append("code_language = %s")
                params.append(filters.code_language)

            if filters.user_intent:
                where_clauses.append("user_intent = %s")
                params.append(filters.user_intent)

            if filters.technical_depth:
                where_clauses.append("technical_depth = %s")
                params.append(filters.technical_depth)

            if filters.response_type:
                where_clauses.append("response_type = %s")
                params.append(filters.response_type)

            if filters.role:
                where_clauses.append("role = %s")
                params.append(filters.role)

            if filters.frustration_indicator is not None:
                where_clauses.append("frustration_indicator = %s")
                params.append(filters.frustration_indicator)

            if filters.urgency_level:
                where_clauses.append("urgency_level = %s")
                params.append(filters.urgency_level)

            if filters.tools_used:
                where_clauses.append("tools_used && %s")
                params.append(filters.tools_used)

            if filters.mcp_tools_used:
                where_clauses.append("mcp_tools_used && %s")
                params.append(filters.mcp_tools_used)

        where_clause = " AND ".join(where_clauses) if where_clauses else "TRUE"

        # Execute vector search
        cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        sql = f"""
            SELECT
                id,
                conversation_id,
                message_index,
                role,
                content,
                contains_code,
                code_language,
                user_intent,
                technical_depth,
                response_type,
                tools_used,
                mcp_tools_used,
                embedding <=> %s::vector AS distance
            FROM message_embeddings
            WHERE {where_clause}
            AND embedding IS NOT NULL
            AND embedding <=> %s::vector < %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        # Params: [query_embedding for each <=> operator, filter params, distance_threshold, top_k]
        all_params = [query_embedding] + params + [query_embedding, distance_threshold, query_embedding, top_k]

        cursor.execute(sql, all_params)
        results = cursor.fetchall()

        # Convert to list of dicts
        output = []
        for row in results:
            result_dict = dict(row)

            # Include conversation context if requested
            if include_context:
                context_cursor = self.conn.cursor(cursor_factory=RealDictCursor)
                context_cursor.execute("""
                    SELECT role, content, message_index
                    FROM message_embeddings
                    WHERE conversation_id = %s
                    AND message_index < %s
                    ORDER BY message_index DESC
                    LIMIT 3
                """, (row['conversation_id'], row['message_index']))

                result_dict['previous_messages'] = list(context_cursor.fetchall())
                context_cursor.close()

            output.append(result_dict)

        cursor.close()
        return output

    def route_query(self, query: str) -> QueryType:
        """
        Determine query type for routing to specialized retrieval strategy

        Simple heuristic-based routing (could be enhanced with LLM classification)
        """
        query_lower = query.lower()

        # Debugging queries
        if any(word in query_lower for word in ['fix', 'fixed', 'solved', 'resolved', 'bug', 'error', 'issue']):
            return QueryType.DEBUGGING

        # Code search queries
        if any(word in query_lower for word in ['code', 'implementation', 'function', 'class', 'snippet']):
            return QueryType.CODE_SEARCH

        # Workflow queries
        if any(word in query_lower for word in ['how to', 'how do', 'how did', 'steps', 'process', 'workflow']):
            return QueryType.WORKFLOW

        # Error recovery queries
        if any(word in query_lower for word in ['went wrong', 'failed', 'error', 'crash', 'broken']):
            return QueryType.ERROR_RECOVERY

        # Default to conceptual
        return QueryType.CONCEPTUAL

    def search_with_routing(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Smart search that routes to appropriate retrieval strategy

        Different query types get different filters and ranking
        """
        query_type = self.route_query(query)

        if query_type == QueryType.DEBUGGING:
            # Debugging: prioritize code + tools
            filters = RetrievalFilters(
                contains_code=True,
                response_type="solution"
            )
            return self.search(query, top_k=top_k, filters=filters)

        elif query_type == QueryType.CODE_SEARCH:
            # Code search: only code messages
            filters = RetrievalFilters(contains_code=True)
            return self.search(query, top_k=top_k, filters=filters)

        elif query_type == QueryType.WORKFLOW:
            # Workflow: include context heavily
            return self.search(query, top_k=top_k, include_context=True)

        elif query_type == QueryType.ERROR_RECOVERY:
            # Error recovery: prioritize frustration indicators
            filters = RetrievalFilters(frustration_indicator=True)
            return self.search(query, top_k=top_k, filters=filters)

        else:
            # Conceptual: no filters, semantic search only
            return self.search(query, top_k=top_k)

    def close(self):
        """Clean up database connection"""
        self.conn.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SPARKII RAG: Stella Retriever Test")
    print("="*80 + "\n")

    # Initialize retriever
    retriever = SparkiiRetriever()

    # Test queries
    test_queries = [
        "How did I fix pandas groupby performance issues?",
        "Show me postgres connection pooling code",
        "What is RAG and how does it work?",
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"🔍 Query: {query}")
        print('='*80)

        results = retriever.search_with_routing(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}] Distance: {result['distance']:.4f}")
            print(f"Role: {result['role']}")
            print(f"Content: {result['content'][:200]}...")
            if result.get('code_language'):
                print(f"Language: {result['code_language']}")

    retriever.close()
    print("\n✅ Done")
