"""RAG 问答服务：编排 LangGraph 工作流并管理会话"""

import time
import json
from typing import AsyncGenerator, Any

from core.graph import RAGState
from core.nodes.query_classifier import classify_query
from core.nodes.retriever import hybrid_retrieve
from core.nodes.generator import llm_generate, llm_generate_stream
from core.nodes.evaluator import evaluate_quality
from core.vectorestore import K12VectorStore
from models.db_models import QARecord, get_session_maker
from utils.logger import logger


class RAGService:
    """RAG 问答服务"""

    def __init__(self, vector_store: K12VectorStore, rag_graph: Any):
        self.vector_store = vector_store
        self.rag_graph = rag_graph

    async def ask(
        self,
        query: str,
        subject: str | None = None,
        grade: str | None = None,
        user_id: str | None = None,
        stream: bool = False,
    ) -> dict:
        """
        执行 RAG 问答流程。

        返回包含 answer、references、latency_ms 等字段的字典。
        """
        start_time = time.time()
        logger.info(f"========== RAG 问答开始 ==========")
        logger.info(f"问题: {query[:100]}")
        logger.info(f"过滤条件: subject={subject}, grade={grade}, user_id={user_id}")

        # 执行 LangGraph 工作流
        initial_state: RAGState = {
            "query": query,
            "subject": subject,
            "grade": grade,
            "complexity": "",
            "retrieved_docs": [],
            "answer": "",
            "evaluation_reason": "",
            "retry_count": 0,
            "max_retries": 2,
        }

        try:
            result = await self.rag_graph.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"RAG 工作流执行失败: {e}")
            return {
                "answer": f"抱歉，回答生成过程中出现错误: {str(e)}",
                "references": [],
                "latency_ms": int((time.time() - start_time) * 1000),
                "complexity": "",
            }

        elapsed = int((time.time() - start_time) * 1000)
        logger.info(f"RAG 流程完成，耗时: {elapsed}ms")

        # 组装引用信息
        references = []
        for doc in result.get("retrieved_docs", []):
            references.append({
                "chunk_id": doc.get("id"),
                "text": doc.get("text", "")[:200],
                "source": doc.get("doc_id", ""),
                "score": round(doc.get("score", 0), 4),
                "subject": doc.get("subject", ""),
                "grade": doc.get("grade", ""),
            })

        answer = result.get("answer", "抱歉，暂时无法回答该问题。")

        # 异步记录问答历史
        record_id = None
        if user_id:
            try:
                record_id = await self._save_qa_record(
                    user_id=user_id,
                    query=query,
                    answer=answer,
                    subject=subject or "",
                    grade=grade or "",
                    complexity=result.get("complexity", "medium"),
                    retrieved_chunks=references[:5],
                    latency_ms=elapsed,
                )
            except Exception as e:
                logger.warning(f"保存问答记录失败: {e}")

        logger.info(f"========== RAG 问答结束 ==========")

        return {
            "answer": answer,
            "references": references,
            "latency_ms": elapsed,
            "complexity": result.get("complexity", "medium"),
            "record_id": record_id,
        }

    async def ask_stream(
        self,
        query: str,
        subject: str | None = None,
        grade: str | None = None,
        user_id: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        流式 RAG 问答，以 SSE 格式逐事件产出：

        event: status  → 状态更新
        event: token   → 回答片段
        event: done    → 完成（含引用元数据）
        """
        start_time = time.time()
        logger.info(f"========== RAG 流式问答开始 ==========")
        logger.info(f"问题: {query[:100]}")
        logger.info(f"过滤条件: subject={subject}, grade={grade}")

        def _sse(event: str, data: dict) -> bytes:
            payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            return payload.encode("utf-8")

        # ---------- 分类 ----------
        yield _sse("status", {"status": "classifying", "message": "正在分析问题复杂度..."})
        complexity = classify_query(query)
        logger.info(f"查询分类: {complexity}")
        yield _sse("status", {"status": "classifying", "message": f"问题分类完成: {complexity}"})

        # ---------- 检索 ----------
        top_k_map = {"simple": 3, "medium": 5, "complex": 8}
        top_k = top_k_map.get(complexity, 5)
        yield _sse("status", {"status": "retrieving", "message": f"正在检索相关知识库 (top_k={top_k})..."})
        try:
            docs = hybrid_retrieve(
                vector_store=self.vector_store,
                query=query,
                complexity=complexity,
                subject=subject,
                grade=grade,
            )
        except Exception as e:
            logger.error(f"检索失败: {e}")
            yield _sse("status", {"status": "error", "message": f"检索失败: {e}"})
            yield _sse("done", {"answer": f"检索失败: {e}", "references": [], "complexity": complexity})
            return

        if not docs:
            yield _sse("status", {"status": "retrieving", "message": "未检索到相关文档"})
            yield _sse("done", {"answer": "未检索到相关文档", "references": [], "complexity": complexity})
            return

        yield _sse("status", {"status": "retrieving", "message": f"检索到 {len(docs)} 篇相关文档"})

        # ---------- 生成（流式） ----------
        yield _sse("status", {"status": "generating", "message": "正在生成回答..."})

        max_retries = 2
        retry_count = 0
        full_answer = ""

        while True:
            # 流式生成
            token_count = 0
            async for token in llm_generate_stream(query=query, context_docs=docs):
                yield _sse("token", {"token": token})
                full_answer += token
                token_count += 1

            logger.info(f"流式生成完成，共 {token_count} 个片段")

            # ---------- 评估 ----------
            yield _sse("status", {"status": "evaluating", "message": "正在评估回答质量..."})
            decision, reason = evaluate_quality(
                query=query,
                answer=full_answer,
                retrieved_docs=docs,
                retry_count=retry_count,
                max_retries=max_retries,
            )

            if decision == "accept":
                logger.info("回答评估通过")
                break
            elif decision == "retry":
                retry_count += 1
                logger.info(f"回答评估不通过，重试第 {retry_count} 次...")
                yield _sse("status", {
                    "status": "retrying",
                    "message": f"回答质量不足，正在重新检索并生成 (重试 {retry_count}/{max_retries})...",
                })
                # 重新检索（扩大范围）
                from core.nodes.retriever import hybrid_retrieve as re_retrieve
                docs = hybrid_retrieve(
                    vector_store=self.vector_store,
                    query=query,
                    complexity="complex",
                    subject=subject,
                    grade=grade,
                )
                yield _sse("status", {
                    "status": "retrying",
                    "message": f"重新检索完成，共 {len(docs)} 篇文档，正在重新生成...",
                })
                full_answer = ""
                continue
            else:
                logger.info("回答评估不通过，放弃重试")
                if not full_answer:
                    full_answer = "未找到相关信息"
                break

        elapsed = int((time.time() - start_time) * 1000)

        # 组装引用
        references = []
        for doc in docs:
            references.append({
                "chunk_id": doc.get("id"),
                "text": doc.get("text", "")[:200],
                "source": doc.get("doc_id", ""),
                "score": round(doc.get("score", 0), 4),
                "subject": doc.get("subject", ""),
                "grade": doc.get("grade", ""),
            })

        # 保存记录
        record_id = None
        if user_id:
            try:
                record_id = await self._save_qa_record(
                    user_id=user_id,
                    query=query,
                    answer=full_answer,
                    subject=subject or "",
                    grade=grade or "",
                    complexity=complexity,
                    retrieved_chunks=references[:5],
                    latency_ms=elapsed,
                )
            except Exception as e:
                logger.warning(f"保存问答记录失败: {e}")

        logger.info(f"========== RAG 流式问答结束 (耗时: {elapsed}ms) ==========")
        yield _sse("done", {
            "answer": full_answer,
            "references": references,
            "latency_ms": elapsed,
            "complexity": complexity,
            "record_id": record_id,
        })

    async def _save_qa_record(
        self,
        user_id: str,
        query: str,
        answer: str,
        subject: str,
        grade: str,
        complexity: str,
        retrieved_chunks: list,
        latency_ms: int,
    ):
        """保存问答记录到数据库"""
        session_maker = get_session_maker()
        async with session_maker() as session:
            record = QARecord(
                user_id=user_id,
                query=query,
                answer=answer,
                subject=subject,
                grade=grade,
                complexity=complexity,
                retrieved_chunks=retrieved_chunks,
                latency_ms=latency_ms,
            )
            session.add(record)
            await session.commit()
            logger.debug(f"问答记录已保存: {record.id}")
            return record.id
