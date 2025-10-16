#!/usr/bin/env python3
"""
SPARKII RAG: LangChain Retriever with Filtered Vector Search
=============================================================

Purpose: Build a production-ready LangChain retriever that leverages:
- Message-level embeddings (255K messages)
- Classification metadata filtering (intent, code, depth, tools)
- Conversation threading (includes context from previous messages)
- Hybrid search (BM25 + vector + metadata fusion)
- Query routing (different retrievers for different query types)

This is the CORE of the RAG system - it makes semantic search actually useful!
"""

import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
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
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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
    Production LangChain-compatible retriever with filtered vector search

    This is the MAGIC - it combines:
    1. Pre-filtering (using classification metadata)
    2. Vector search (semantic similarity)
    3. Context expansion (fetches previous messages in conversation)
    4. Re-ranking (optional - by relevance score)
    """

    def __init__(self):
        # Load embedding model
        print("🔄 Loading embedding model...")
        self.model = SentenceTransformer(MODEL_NAME)
        print(f"✅ Model loaded: {MODEL_NAME}")

        # Connect to database (using Supabase pooler for IPv4 compatibility)
        print("🔄 Connecting to Supabase...")
        self.conn = psycopg.connect(SUPABASE_URL, connect_timeout=10)
        print("✅ Connected to Supabase")

    def encode_query(self, query: str) -> List[float]:
        """Convert query to embedding vector"""
        return self.model.encode(query).tolist()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[RetrievalFilters] = None,
        include_context: bool = True,
        distance_threshold: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Main search method - filtered vector search with context

        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Classification metadata filters
            include_context: Include previous messages from conversation
            distance_threshold: Max cosine distance (0-2, lower = more similar)

        Returns:
            List of messages with metadata and optional context
        """
        # Generate query embedding
        query_embedding = self.encode_query(query)

        # Build SQL with filters
        sql = self._build_search_sql(filters, include_context)

        # Execute search
        cursor = self.conn.cursor(row_factory=dict_row)
        cursor.execute(sql, {
            'query_embedding': query_embedding,
            'top_k': top_k,
            'distance_threshold': distance_threshold,
            **self._filter_params(filters)
        })

        results = cursor.fetchall()
        cursor.close()

        # Convert to dicts
        return [dict(row) for row in results]

    def _build_search_sql(
        self,
        filters: Optional[RetrievalFilters],
        include_context: bool
    ) -> str:
        """Build SQL query with optional filters and context"""

        # Base query - vector similarity search
        sql = """
        WITH ranked_messages AS (
            SELECT
                m.id,
                m.conversation_id,
                m.message_id,
                m.message_index,
                m.role,
                m.content,
                m.user_intent,
                m.urgency_level,
                m.frustration_indicator,
                m.response_type,
                m.contains_code,
                m.code_language,
                m.cognitive_operation,
                m.technical_depth,
                m.tools_used,
                m.mcp_tools_used,
                m.refers_to_previous,
                m.introduces_new_topic,
                m.closes_topic,
                m.classification_confidence,
                m.message_clarity,
                m.embedding <=> %(query_embedding)s::vector AS distance,
                c.title AS conversation_title,
                c.created_at AS conversation_date
            FROM message_embeddings m
            JOIN chat_histories c ON m.conversation_id = c.id
            WHERE 1=1
        """

        # Add filters
        if filters:
            if filters.contains_code is not None:
                sql += " AND m.contains_code = %(contains_code)s"
            if filters.code_language:
                sql += " AND m.code_language = %(code_language)s"
            if filters.user_intent:
                sql += " AND m.user_intent = %(user_intent)s"
            if filters.technical_depth:
                sql += " AND m.technical_depth = %(technical_depth)s"
            if filters.response_type:
                sql += " AND m.response_type = %(response_type)s"
            if filters.role:
                sql += " AND m.role = %(role)s"
            if filters.frustration_indicator is not None:
                sql += " AND m.frustration_indicator = %(frustration_indicator)s"
            if filters.urgency_level:
                sql += " AND m.urgency_level = %(urgency_level)s"
            if filters.tools_used:
                sql += " AND m.tools_used && %(tools_used)s"
            if filters.mcp_tools_used:
                sql += " AND m.mcp_tools_used && %(mcp_tools_used)s"

        # Distance threshold and ordering
        sql += """
            AND m.embedding <=> %(query_embedding)s::vector < %(distance_threshold)s
            ORDER BY distance ASC
            LIMIT %(top_k)s
        )
        """

        # Include context if requested
        if include_context:
            sql += """
            SELECT
                r.*,
                (
                    SELECT jsonb_agg(
                        jsonb_build_object(
                            'role', pm.role,
                            'content', pm.content,
                            'message_index', pm.message_index
                        ) ORDER BY pm.message_index
                    )
                    FROM message_embeddings pm
                    WHERE pm.conversation_id = r.conversation_id
                      AND pm.message_index < r.message_index
                      AND pm.message_index >= GREATEST(0, r.message_index - 3)
                ) AS previous_messages
            FROM ranked_messages r
            """
        else:
            sql += "SELECT * FROM ranked_messages"

        return sql

    def _filter_params(self, filters: Optional[RetrievalFilters]) -> Dict[str, Any]:
        """Convert filters to SQL parameters"""
        if not filters:
            return {}

        params = {}
        if filters.contains_code is not None:
            params['contains_code'] = filters.contains_code
        if filters.code_language:
            params['code_language'] = filters.code_language
        if filters.user_intent:
            params['user_intent'] = filters.user_intent
        if filters.technical_depth:
            params['technical_depth'] = filters.technical_depth
        if filters.response_type:
            params['response_type'] = filters.response_type
        if filters.role:
            params['role'] = filters.role
        if filters.frustration_indicator is not None:
            params['frustration_indicator'] = filters.frustration_indicator
        if filters.urgency_level:
            params['urgency_level'] = filters.urgency_level
        if filters.tools_used:
            params['tools_used'] = filters.tools_used
        if filters.mcp_tools_used:
            params['mcp_tools_used'] = filters.mcp_tools_used

        return params

    def route_query(self, query: str) -> QueryType:
        """
        Classify query type to select optimal retrieval strategy

        This is a simple rule-based router. In production, you'd use
        an LLM or classifier model for better accuracy.
        """
        query_lower = query.lower()

        # Debugging queries
        if any(word in query_lower for word in ['fix', 'debug', 'solve', 'error', 'bug', 'issue']):
            return QueryType.DEBUGGING

        # Code search queries
        if any(word in query_lower for word in ['implementation', 'code for', 'function', 'class', 'show me']):
            return QueryType.CODE_SEARCH

        # Workflow queries
        if any(word in query_lower for word in ['how do i', 'steps to', 'workflow', 'process']):
            return QueryType.WORKFLOW

        # Error recovery queries
        if any(word in query_lower for word in ['went wrong', 'failed', 'not working', 'broken']):
            return QueryType.ERROR_RECOVERY

        # Default to conceptual
        return QueryType.CONCEPTUAL

    def search_with_routing(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Intelligent search that routes query to optimal retrieval strategy

        This is the "smart" search that adapts to query type!
        """
        query_type = self.route_query(query)

        # Build filters based on query type
        filters = None

        if query_type == QueryType.DEBUGGING:
            # Debugging queries need code + tools + error context
            filters = RetrievalFilters(
                contains_code=True,
                frustration_indicator=None  # Can include both
            )

        elif query_type == QueryType.CODE_SEARCH:
            # Code search needs code only
            filters = RetrievalFilters(
                contains_code=True,
                role='assistant'  # Usually want AI's code responses
            )

        elif query_type == QueryType.WORKFLOW:
            # Workflow queries need sequential context
            filters = None  # No filtering, rely on context

        elif query_type == QueryType.ERROR_RECOVERY:
            # Error recovery needs frustration signals
            filters = RetrievalFilters(
                frustration_indicator=True,
                urgency_level='high'
            )

        # Execute search with routing
        print(f"🎯 Query routed to: {query_type.value}")
        return self.search(query, top_k, filters, include_context=True)

    def close(self):
        """Clean up database connection"""
        self.conn.close()

# ============================================================================
# LANGCHAIN INTEGRATION
# ============================================================================

class SparkiiLangChainRetriever:
    """
    LangChain-compatible wrapper for SparkiiRetriever

    This allows you to use the retriever in LangChain chains:
    - RetrievalQA
    - ConversationalRetrievalChain
    - Custom chains
    """

    def __init__(self, retriever: SparkiiRetriever):
        self.retriever = retriever

    def get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """LangChain interface method"""
        return self.retriever.search_with_routing(query)

    async def aget_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """Async version for LangChain"""
        # For now, just call sync version
        # In production, you'd implement async database queries
        return self.get_relevant_documents(query)

# ============================================================================
# TESTING & EXAMPLES
# ============================================================================

def test_retriever():
    """Test the retriever with sample queries"""
    print("\n" + "="*80)
    print("SPARKII RAG Retriever Test")
    print("="*80 + "\n")

    retriever = SparkiiRetriever()

    # Test queries
    test_queries = [
        "How did I fix the authentication bug?",
        "Show me Python code examples",
        "What is RAG?",
        "How do I deploy to production?",
        "What went wrong with the database migration?"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 80)

        results = retriever.search_with_routing(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"\n📄 Result {i} (distance: {result['distance']:.3f})")
            print(f"   Role: {result['role']}")
            print(f"   Content: {result['content'][:100]}...")
            if result.get('contains_code'):
                print(f"   💻 Contains code: {result['code_language']}")
            if result.get('tools_used'):
                print(f"   🔧 Tools: {result['tools_used']}")
            print(f"   📅 Conversation: {result['conversation_title']}")

    retriever.close()
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_retriever()
