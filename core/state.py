"""Shared state definitions for the RAG workflow."""

from typing import TypedDict


MAX_ROUNDS = 10


class RAGState(TypedDict):
    """RAG 流程的全局状态"""

    query: str
    subject: str | None
    grade: str | None
    intent: str
    complexity: str
    retrieved_docs: list
    answer: str
    evaluation_reason: str
    evaluation_decision: str
    retry_count: int
    max_retries: int
    conversation_history: list[dict]
    _queue_id: str
