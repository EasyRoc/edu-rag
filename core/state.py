"""Shared state definitions for the RAG workflow."""

from typing import TypedDict


MAX_ROUNDS = 10


class RAGState(TypedDict):
    """RAG 流程的全局状态"""

    query: str
    subject: str | None
    grade: str | None
    session_id: str
    intent: str
    complexity: str
    retrieved_docs: list
    answer: str
    retry_count: int
    max_retries: int
    conversation_history: list[dict]
    retrieval_plan: dict
    retrieval_attempts: list[dict]
    retrieval_metrics: dict
    retrieval_decision: dict
    abstain_reason: str
    retrieval_latency_ms: float
    rerank_latency_ms: float
    reranker_available: bool
    sub_queries: list[str]
    _queue_id: str
