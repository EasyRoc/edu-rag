"""RAG 问答服务：通过 LangGraph 工作流编排问答流程"""

import time
import json
import uuid
import asyncio
from typing import AsyncGenerator, Any

from core.graph import RAGState
from core.stream_queue import _registry as _stream_queues
from core.vectorestore import K12VectorStore
from models.db_models import QARecord, get_session_maker
from utils.logger import logger


class RAGService:
    """RAG 问答服务，所有问答流程均通过 LangGraph 工作流编排"""

    def __init__(self, vector_store: K12VectorStore, rag_graph: Any):
        self.vector_store = vector_store
        self.rag_graph = rag_graph

    def _build_initial_state(
        self, query: str, subject: str | None, grade: str | None, user_id: str | None
    ) -> dict:
        return {
            "query": query,
            "subject": subject,
            "grade": grade,
            "intent": "",
            "complexity": "",
            "retrieved_docs": [],
            "answer": "",
            "evaluation_reason": "",
            "evaluation_decision": "",
            "retry_count": 0,
            "max_retries": 2,
            "_queue_id": "",
        }

    async def ask(
        self,
        query: str,
        subject: str | None = None,
        grade: str | None = None,
        user_id: str | None = None,
        stream: bool = False,
    ) -> dict:
        """
        执行 RAG 问答流程（非流式）。

        通过 LangGraph 工作流完成意图识别 → 检索 → 生成 → 评估 → 纠错。
        """
        start_time = time.time()
        logger.info(f"========== RAG 问答开始 ==========")
        logger.info(f"问题: {query[:100]}")
        logger.info(f"过滤条件: subject={subject}, grade={grade}, user_id={user_id}")

        initial_state = self._build_initial_state(query, subject, grade, user_id)
        config = {"configurable": {"thread_id": user_id or "default"}}

        try:
            final_state = await self.rag_graph.ainvoke(initial_state, config)
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
        for doc in final_state.get("retrieved_docs", []):
            references.append({
                "index": len(references) + 1,
                "chunk_id": doc.get("id"),
                "text": doc.get("text", "")[:200],
                "source_file": doc.get("source_file") or doc.get("doc_id") or "未知来源",
                "page": doc.get("page", 0),
                "chapter": doc.get("chapter", ""),
                "score": round(doc.get("score", 0), 4),
                "subject": doc.get("subject", ""),
                "grade": doc.get("grade", ""),
            })
            # 保留 source 字段向后兼容
            references[-1]["source"] = doc.get("doc_id", "")

        answer = final_state.get("answer", "抱歉，暂时无法回答该问题。")

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
                    complexity=final_state.get("complexity", "medium"),
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
            "complexity": final_state.get("complexity", "medium"),
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
        流式 RAG 问答，通过 LangGraph 工作流编排，以 SSE 格式逐事件产出。

        通过 asyncio.Queue 在 graph 节点与服务之间传递 token，
        实现边生成边推送的流式效果，同时 graph 全流程在后台运行。

        event: status  → 状态更新
        event: token   → 回答片段
        event: done    → 完成（含完整回答及引用元数据）
        """
        start_time = time.time()
        logger.info(f"========== RAG 流式问答开始 ==========")
        logger.info(f"问题: {query[:100]}")
        logger.info(f"过滤条件: subject={subject}, grade={grade}")

        def _sse(event: str, data: dict) -> bytes:
            payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            return payload.encode("utf-8")

        # 创建 token 队列，通过全局注册表传递（避免 Queue 直接放入 LangGraph state 导致深拷贝失败）
        queue_id = str(uuid.uuid4())
        stream_queue: asyncio.Queue = asyncio.Queue()
        _stream_queues[queue_id] = stream_queue

        initial_state = self._build_initial_state(query, subject, grade, user_id)
        initial_state["_queue_id"] = queue_id
        config = {"configurable": {"thread_id": user_id or "default"}}

        full_answer = ""
        final_state = {}
        graph_error = None

        async def _run_graph():
            """后台运行 LangGraph 工作流，状态通过 astream 收集"""
            nonlocal final_state, graph_error
            try:
                async for state in self.rag_graph.astream(
                    initial_state, config, stream_mode="values"
                ):
                    final_state = state
            except Exception as e:
                graph_error = e
                logger.error(f"RAG 后台工作流异常: {e}")

        # 启动后台 graph 任务
        graph_task = asyncio.create_task(_run_graph())

        try:
            # 先发送初始状态
            yield _sse("status", {
                "status": "classifying",
                "message": "正在分析问题...",
            })

            # 从队列中读取 token，直到收到 None 标记
            while True:
                token = await stream_queue.get()
                if token is None:
                    break
                full_answer += token
                # 首次收到 token 时，发送"生成中"状态
                if len(full_answer) == len(token):
                    yield _sse("status", {
                        "status": "generating",
                        "message": "正在生成回答...",
                    })
                yield _sse("token", {"token": token})

            # 等待 graph 完全结束
            await graph_task

        except asyncio.CancelledError:
            graph_task.cancel()
            logger.warning("流式请求被取消")
            return
        finally:
            _stream_queues.pop(queue_id, None)

        # 计算最终回答（处理不同分支）
        if graph_error:
            if not full_answer:
                full_answer = "抱歉，回答生成过程中出现错误，请稍后重试。"
            final_answer = full_answer
            references = []
            final_complexity = ""
        elif final_state.get("intent") == "chitchat":
            final_answer = final_state.get("answer", full_answer)
            references = []
            final_complexity = "simple"
        else:
            final_answer = final_state.get("answer", full_answer)
            references = []
            for doc in final_state.get("retrieved_docs", []):
                references.append({
                    "index": len(references) + 1,
                    "chunk_id": doc.get("id"),
                    "text": doc.get("text", "")[:200],
                    "source_file": doc.get("source_file") or doc.get("doc_id") or "未知来源",
                    "page": doc.get("page", 0),
                    "chapter": doc.get("chapter", ""),
                    "score": round(doc.get("score", 0), 4),
                    "subject": doc.get("subject", ""),
                    "grade": doc.get("grade", ""),
                })
                # 保留 source 字段向后兼容
                references[-1]["source"] = doc.get("doc_id", "")
            final_complexity = final_state.get("complexity", "medium")

        elapsed = int((time.time() - start_time) * 1000)

        # 保存数据库记录
        record_id = None
        if user_id:
            try:
                record_id = await self._save_qa_record(
                    user_id=user_id,
                    query=query,
                    answer=final_answer,
                    subject=subject or "",
                    grade=grade or "",
                    complexity=final_complexity,
                    retrieved_chunks=references[:5],
                    latency_ms=elapsed,
                )
            except Exception as e:
                logger.warning(f"保存问答记录失败: {e}")

        logger.info(f"========== RAG 流式问答结束 (耗时: {elapsed}ms) ==========")
        yield _sse("done", {
            "answer": final_answer,
            "references": references,
            "latency_ms": elapsed,
            "complexity": final_complexity,
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
