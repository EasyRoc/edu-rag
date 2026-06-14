"""K12 RAG 的 LangGraph 编排入口。

图中只保存可序列化状态；向量库、重排器等运行时对象通过闭包注入节点，
避免被 LangGraph checkpointer 持久化。
"""

from __future__ import annotations

import time
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from config import settings
from core.nodes.chitchat import chitchat_node
from core.nodes.generator import generate_sub_answers, llm_generate_stream, synthesize_final_answer
from core.nodes.query_classifier import classify_intent_async, classify_query_with_fallback
from core.nodes.retriever import build_retry_plan, hybrid_retrieve
from core.reranker import CrossEncoderReranker, RerankerUnavailableError
from core.retrieval_quality import evaluate_retrieval_gate
from core.state import MAX_ROUNDS, RAGState
from core.stream_queue import stream_queues
from core.vectorestore import K12VectorStore
from utils.logger import logger


ABSTAIN_ANSWER = "抱歉，我暂时没有检索到足够可靠的资料来回答这个问题。你可以补充教材范围、年级或更具体的问题。"


async def finalize_node(state: RAGState) -> dict:
    """收尾节点：把本轮用户问题与助手回答写入会话历史。"""
    history = list(state.get("conversation_history", []))
    history.extend(
        [
            {"role": "user", "content": state["query"]},
            {"role": "assistant", "content": state.get("answer", "")},
        ]
    )
    trimmed = history[-MAX_ROUNDS * 2 :]
    logger.debug("finalize: 会话历史裁剪为 %d 条消息", len(trimmed))
    return {"conversation_history": trimmed}


async def classify_node(state: RAGState) -> dict:
    """意图与复杂度分类节点。非教育问题直接转闲聊分支。"""
    intent = await classify_intent_async(state["query"])
    complexity = await classify_query_with_fallback(state["query"]) if intent == "educational" else "simple"
    logger.info(
        "classify: intent=%s, complexity=%s, query=%s",
        intent,
        complexity,
        state["query"][:50],
    )
    return {"intent": intent, "complexity": complexity}

async def retrieve_node(state: RAGState, vector_store: K12VectorStore) -> dict:
    """候选召回节点：只负责召回，不在这里做质量判断。"""
    started = time.perf_counter()
    docs, sub_queries = await hybrid_retrieve(
        vector_store=vector_store,
        query=state["query"],
        complexity=state["complexity"],
        intent=state.get("intent", "educational"),
        subject=state.get("subject"),
        grade=state.get("grade"),
        retrieval_plan=state.get("retrieval_plan"),
        candidate_top_k=settings.RETRIEVAL_CANDIDATE_TOP_K,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    logger.info(
        "retrieve: plan=%s, candidates=%d, sub_queries=%d, latency_ms=%.3f",
        state.get("retrieval_plan", {}).get("strategy", "initial"),
        len(docs),
        len(sub_queries),
        latency_ms,
    )
    return {
        "retrieved_docs": docs,
        "retrieval_latency_ms": latency_ms,
        "sub_queries": sub_queries,
    }


async def rerank_node(state: RAGState, reranker: CrossEncoderReranker) -> dict:
    """本地 CrossEncoder 重排节点。失败时交给门控决定是否拒答或观察放行。"""
    started = time.perf_counter()
    docs = state.get("retrieved_docs", [])
    try:
        if _should_use_two_stage_rerank(
            state.get("complexity", "medium"),
            state.get("sub_queries", []),
            settings.ENABLE_DEEP_COMPLEX_MODE,
        ):
            docs = await _two_stage_rerank(
                state["query"],
                docs,
                state.get("sub_queries", []),
                reranker,
            )
        else:
            docs = await reranker.rerank(state["query"], docs)
        available = True
    except RerankerUnavailableError as exc:
        logger.warning("本地重排不可用: %s", exc)
        docs = list(docs)
        available = False
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    logger.info(
        "rerank: available=%s, docs=%d, top1=%.4f, latency_ms=%.3f",
        available,
        len(docs),
        docs[0].get("rerank_score", 0.0) if docs else 0.0,
        latency_ms,
    )
    return {
        "retrieved_docs": docs,
        "reranker_available": available,
        "rerank_latency_ms": latency_ms,
    }


def _should_use_two_stage_rerank(
    complexity: str,
    sub_queries: list[str],
    deep_mode_enabled: bool,
) -> bool:
    """复杂问题且存在多个子问题时启用两阶段重排。"""
    return complexity == "complex" and len(sub_queries) >= 2 and deep_mode_enabled


async def _two_stage_rerank(
    original_query: str,
    docs: list[dict],
    sub_queries: list[str],
    reranker: CrossEncoderReranker,
) -> list[dict]:
    """两阶段重排：子问题独立 rerank 后，再用原问题做最终 rerank。"""
    if not docs:
        return []

    grouped: dict[str, list[dict]] = {}
    no_source_key = "__no_source__"
    for doc in docs:
        source = str(doc.get("source_sub_query", ""))
        matched = [item for item in sub_queries if item and item in source]
        if not matched and source:
            matched = [source]
        if not matched:
            matched = [no_source_key]
        for key in matched:
            grouped.setdefault(key, []).append(doc)

    stage1_results: list[dict] = []
    for source_query, group_docs in grouped.items():
        query = original_query if source_query == no_source_key else source_query
        reranked = await reranker.rerank(query, group_docs)
        stage1_results.extend(reranked[: settings.SUB_RERANK_TOP_K])

    seen_ids: set[str] = set()
    merged: list[dict] = []
    for doc in sorted(stage1_results, key=lambda item: item.get("rerank_score", 0.0), reverse=True):
        doc_id = str(doc.get("id") or doc.get("chunk_id") or doc.get("text", ""))
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        merged.append(doc)

    logger.info(
        "两阶段重排: stage1_groups=%d, stage1_docs=%d, merged=%d",
        len(grouped),
        len(stage1_results),
        len(merged),
    )
    return await reranker.rerank(original_query, merged)


async def retrieval_gate_node(state: RAGState) -> dict:
    """统一检索门控节点：根据重排分数决定生成、重试或拒答。"""
    decision = evaluate_retrieval_gate(
        state.get("retrieved_docs", []),
        retry_count=state.get("retry_count", 0),
        max_retries=state.get("max_retries", settings.MAX_RETRIES),
        reranker_available=state.get("reranker_available", False),
        complexity=state.get("complexity", "medium"),
        sub_queries=state.get("sub_queries", []),
    )
    attempt = {
        "retry_count": state.get("retry_count", 0),
        "plan": state.get("retrieval_plan", {}),
        "candidate_count": len(state.get("retrieved_docs", [])),
        "metrics": decision["metrics"],
        "decision": decision["action"],
        "reason_codes": decision["reason_codes"],
        "retrieval_latency_ms": state.get("retrieval_latency_ms", 0.0),
        "rerank_latency_ms": state.get("rerank_latency_ms", 0.0),
    }
    logger.info(
        "retrieval_gate: action=%s, reasons=%s, retry=%d/%d, metrics=%s",
        decision["action"],
        ",".join(decision["reason_codes"]),
        state.get("retry_count", 0),
        state.get("max_retries", settings.MAX_RETRIES),
        decision["metrics"],
    )
    return {
        "retrieval_metrics": decision["metrics"],
        "retrieval_decision": decision,
        "retrieval_attempts": [*state.get("retrieval_attempts", []), attempt],
    }


async def retry_planner_node(state: RAGState) -> dict:
    """纠正检索规划节点：根据门控原因选择下一轮检索策略。"""
    next_retry = state.get("retry_count", 0) + 1
    plan = await build_retry_plan(
        query=state["query"],
        next_retry_count=next_retry,
        decision=state.get("retrieval_decision", {}),
        complexity=state.get("complexity", "medium"),
        sub_queries=state.get("sub_queries", []),
    )
    logger.info("retry_planner: retry=%d, plan=%s", next_retry, plan)
    return {"retry_count": next_retry, "retrieval_plan": plan}


async def generate_node(state: RAGState) -> dict:
    """答案生成节点。只有门控通过的上下文才会进入这里。"""
    full_answer = ""
    queue_id = state.get("_queue_id")
    all_docs = state.get("retrieved_docs", [])
    sub_queries = state.get("sub_queries", [])

    if _should_use_synthesis(
        state.get("complexity", "medium"),
        sub_queries,
        settings.ENABLE_DEEP_COMPLEX_MODE,
    ):
        docs = all_docs[: settings.COMPLEX_CONTEXT_TOP_K]
        sub_docs_map = _group_docs_by_sub_query(all_docs, sub_queries)
        sub_answers = await generate_sub_answers(sub_queries, sub_docs_map)
        full_answer = await synthesize_final_answer(state["query"], sub_answers, docs)
        await stream_queues.emit(queue_id, full_answer)
        logger.info(
            "generate_complex: sub_queries=%d, context_docs=%d, answer_chars=%d",
            len(sub_queries),
            len(docs),
            len(full_answer),
        )
        return {"answer": full_answer, "retrieved_docs": docs}

    docs = all_docs[: settings.GENERATION_CONTEXT_TOP_K]
    async for token in llm_generate_stream(
        query=state["query"],
        context_docs=docs,
        conversation_history=state.get("conversation_history", []),
    ):
        full_answer += token
        await stream_queues.emit(queue_id, token)
    logger.info("generate: context_docs=%d, answer_chars=%d", len(docs), len(full_answer))
    return {"answer": full_answer, "retrieved_docs": docs}


def _should_use_synthesis(
    complexity: str,
    sub_queries: list[str],
    deep_mode_enabled: bool,
) -> bool:
    """复杂问题且存在多个子问题时，启用子答案合成路径。"""
    return complexity == "complex" and len(sub_queries) >= 2 and deep_mode_enabled


def _group_docs_by_sub_query(docs: list[dict], sub_queries: list[str]) -> dict[str, list[dict]]:
    """按子问题来源给检索片段分组，缺失来源时用全局上下文兜底。"""
    groups = {item: [] for item in sub_queries if item}
    if not groups:
        return {}

    for doc in docs:
        source = str(doc.get("source_sub_query", ""))
        matched = [item for item in groups if item in source]
        if source and source in groups and source not in matched:
            matched.append(source)
        for item in matched:
            groups[item].append(doc)

    fallback_docs = docs[: settings.SUB_RERANK_TOP_K]
    for item, group_docs in groups.items():
        if not group_docs:
            groups[item] = fallback_docs
    return groups


async def abstain_node(state: RAGState) -> dict:
    """拒答节点：低置信检索不进入 LLM 生成，避免把弱证据包装成答案。"""
    decision = state.get("retrieval_decision", {})
    logger.info("abstain: reasons=%s", ",".join(decision.get("reason_codes", [])))
    return {
        "answer": ABSTAIN_ANSWER,
        "retrieved_docs": [],
        "abstain_reason": ",".join(decision.get("reason_codes", [])),
    }


def _route_by_gate(state: RAGState) -> Literal["accept", "retry", "abstain"]:
    """条件边：读取门控节点的结构化动作。"""
    return state.get("retrieval_decision", {}).get("action", "abstain")


def build_rag_graph(
    vector_store: K12VectorStore,
    reranker: CrossEncoderReranker | None = None,
    *,
    checkpointer=None,
):
    """构建 RAG 图，并通过闭包注入不可序列化的运行时依赖。"""
    reranker = reranker or CrossEncoderReranker()
    logger.info(
        "构建 RAG Graph: candidate_top_k=%d, context_top_k=%d, max_retries=%d",
        settings.RETRIEVAL_CANDIDATE_TOP_K,
        settings.GENERATION_CONTEXT_TOP_K,
        settings.MAX_RETRIES,
    )

    async def retrieve_with_store(state: RAGState) -> dict:
        return await retrieve_node(state, vector_store)

    async def rerank_with_model(state: RAGState) -> dict:
        return await rerank_node(state, reranker)

    workflow = StateGraph(RAGState)
    workflow.add_node("classify", classify_node)
    workflow.add_node("retrieve", retrieve_with_store)
    workflow.add_node("rerank", rerank_with_model)
    workflow.add_node("retrieval_gate", retrieval_gate_node)
    workflow.add_node("retry_planner", retry_planner_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("abstain", abstain_node)
    workflow.add_node("chitchat", chitchat_node)
    workflow.add_node("finalize", finalize_node)
    workflow.set_entry_point("classify")
    workflow.add_conditional_edges(
        "classify",
        lambda state: "retrieve" if state.get("intent") == "educational" else "chitchat",
        {"retrieve": "retrieve", "chitchat": "chitchat"},
    )
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "retrieval_gate")
    workflow.add_conditional_edges(
        "retrieval_gate",
        _route_by_gate,
        {"accept": "generate", "retry": "retry_planner", "abstain": "abstain"},
    )
    workflow.add_edge("retry_planner", "retrieve")
    workflow.add_edge("generate", "finalize")
    workflow.add_edge("abstain", "finalize")
    workflow.add_edge("chitchat", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile(checkpointer=MemorySaver() if checkpointer is None else checkpointer)
