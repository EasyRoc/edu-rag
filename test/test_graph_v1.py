"""Graph routing, session isolation and SSE lifecycle tests for RAG V1."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FakeStore:
    def __init__(self, docs_by_call: list[list[dict]]):
        self.docs_by_call = docs_by_call
        self.calls = []

    def hybrid_search(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.docs_by_call) - 1)
        return [dict(doc) for doc in self.docs_by_call[index]]


class FakeReranker:
    async def rerank(self, query, docs):
        ranked = []
        for doc in docs:
            item = dict(doc)
            item["rerank_raw_score"] = item["quality"]
            item["rerank_score"] = item["quality"]
            ranked.append(item)
        return sorted(ranked, key=lambda doc: doc["rerank_score"], reverse=True)


async def fake_generate(**kwargs):
    yield "可靠回答"


class RetryPlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_retry_uses_original_query_and_three_unique_variants(self):
        from core.nodes.retriever import build_retry_plan

        with patch(
            "core.nodes.retriever.generate_query_variants",
            new=AsyncMock(return_value=["改写一", "改写一", "改写二", "改写三", "改写四"]),
        ):
            plan = await build_retry_plan(
                query="原问题",
                next_retry_count=1,
                decision={"suggested_strategy": "hyde"},
            )

        self.assertEqual(plan["strategy"], "query_variants")
        self.assertEqual(plan["queries"], ["原问题", "改写一", "改写二", "改写三"])

    async def test_second_retry_uses_gate_suggestion(self):
        from core.nodes.retriever import build_retry_plan

        plan = await build_retry_plan(
            query="原问题",
            next_retry_count=2,
            decision={"suggested_strategy": "step_back"},
        )

        self.assertEqual(plan, {"strategy": "step_back", "queries": ["原问题"]})

    async def test_complex_retry_uses_sub_query_repair_plan(self):
        from core.nodes.retriever import build_retry_plan

        decision = {
            "suggested_strategy": "hyde",
            "suggested_plan": {
                "strategy": "complex_repair",
                "subqueries": [
                    {"query": "子问题A", "status": "covered", "repair": "direct"},
                    {"query": "子问题B", "status": "missing", "repair": "hyde"},
                ],
            },
        }

        plan = await build_retry_plan(
            query="复杂原问题",
            next_retry_count=1,
            decision=decision,
            complexity="complex",
            sub_queries=["子问题A", "子问题B"],
        )

        self.assertEqual(plan["strategy"], "complex_repair")
        self.assertEqual(plan["sub_queries"], ["子问题A", "子问题B"])
        repairs = {item["query"]: item["repair"] for item in plan["subqueries"]}
        self.assertEqual(repairs, {"子问题A": "direct", "子问题B": "hyde"})

    async def test_complex_retry_retrieve_preserves_sub_queries_and_sources(self):
        from core.nodes.retriever import hybrid_retrieve

        store = FakeStore(
            [
                [{"id": 1, "doc_id": "a", "text": "A", "score": 0.9}],
                [{"id": 2, "doc_id": "b", "text": "B", "score": 0.8}],
            ]
        )
        plan = {
            "strategy": "complex_repair",
            "sub_queries": ["子问题A", "子问题B"],
            "subqueries": [
                {"query": "子问题A", "status": "covered", "repair": "direct"},
                {"query": "子问题B", "status": "covered", "repair": "direct"},
            ],
        }

        docs, sub_queries = await hybrid_retrieve(
            vector_store=store,
            query="复杂原问题",
            complexity="complex",
            retrieval_plan=plan,
            candidate_top_k=10,
        )

        self.assertEqual(sub_queries, ["子问题A", "子问题B"])
        self.assertEqual({doc["source_sub_query"] for doc in docs}, {"子问题A", "子问题B"})


class GraphRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_high_quality_docs_reach_generate(self):
        from core.graph import build_rag_graph

        store = FakeStore([[{"id": 1, "doc_id": "doc", "text": "资料", "quality": 0.9}]])
        graph = build_rag_graph(store, FakeReranker(), checkpointer=False)

        with (
            patch("core.graph.classify_intent_async", new=AsyncMock(return_value="educational")),
            patch("core.graph.classify_query_with_fallback", new=AsyncMock(return_value="simple")),
            patch("core.graph.llm_generate_stream", new=fake_generate),
        ):
            final_state = await graph.ainvoke(_initial_state(max_retries=0))

        self.assertEqual(final_state["answer"], "可靠回答")
        self.assertEqual(final_state["retrieval_decision"]["action"], "accept")

    async def test_low_quality_docs_abstain_without_generate(self):
        from core.graph import ABSTAIN_ANSWER, build_rag_graph

        store = FakeStore([[{"id": 1, "doc_id": "doc", "text": "不可靠资料", "quality": 0.2}]])
        graph = build_rag_graph(store, FakeReranker(), checkpointer=False)
        generator = AsyncMock(side_effect=AssertionError("generator must not run"))

        with (
            patch("core.graph.classify_intent_async", new=AsyncMock(return_value="educational")),
            patch("core.graph.classify_query_with_fallback", new=AsyncMock(return_value="simple")),
            patch("core.graph.llm_generate_stream", new=generator),
        ):
            final_state = await graph.ainvoke(_initial_state(max_retries=0))

        self.assertEqual(final_state["answer"], ABSTAIN_ANSWER)
        self.assertEqual(final_state["retrieved_docs"], [])
        self.assertEqual(final_state["retrieval_decision"]["action"], "abstain")
        generator.assert_not_called()

    async def test_retry_can_recover_before_generation(self):
        from core.graph import build_rag_graph

        store = FakeStore(
            [
                [{"id": 1, "doc_id": "weak", "text": "弱资料", "quality": 0.2}],
                [{"id": 2, "doc_id": "strong", "text": "可靠资料", "quality": 0.9}],
            ]
        )
        graph = build_rag_graph(store, FakeReranker(), checkpointer=False)

        with (
            patch("core.graph.classify_intent_async", new=AsyncMock(return_value="educational")),
            patch("core.graph.classify_query_with_fallback", new=AsyncMock(return_value="simple")),
            patch("core.graph.llm_generate_stream", new=fake_generate),
            patch(
                "core.nodes.retriever.generate_query_variants",
                new=AsyncMock(return_value=["改写问题"]),
            ),
        ):
            final_state = await graph.ainvoke(_initial_state(max_retries=1))

        self.assertEqual(final_state["answer"], "可靠回答")
        self.assertEqual(final_state["retry_count"], 1)
        self.assertEqual(final_state["retrieval_decision"]["action"], "accept")
        self.assertGreaterEqual(len(store.calls), 2)

    async def test_complex_retry_keeps_synthesis_path_after_recovery(self):
        from core.graph import build_rag_graph

        store = FakeStore(
            [
                [{"id": 1, "doc_id": "a1", "text": "A 初始证据", "score": 0.8, "quality": 0.8}],
                [{"id": 2, "doc_id": "b1", "text": "B 初始弱证据", "score": 0.1, "quality": 0.1}],
                [{"id": 3, "doc_id": "a2", "text": "A 修复证据", "score": 0.8, "quality": 0.8}],
                [{"id": 4, "doc_id": "b2", "text": "B 修复证据", "score": 0.75, "quality": 0.75}],
                [{"id": 5, "doc_id": "b3", "text": "B HyDE 修复证据", "score": 0.72, "quality": 0.72}],
            ]
        )
        graph = build_rag_graph(store, FakeReranker(), checkpointer=False)

        with (
            patch("core.graph.classify_intent_async", new=AsyncMock(return_value="educational")),
            patch("core.graph.classify_query_with_fallback", new=AsyncMock(return_value="complex")),
            patch(
                "core.nodes.retriever.decompose_query",
                new=AsyncMock(return_value=["子问题A", "子问题B"]),
            ),
            patch(
                "core.nodes.retriever.generate_hypothetical_answer",
                new=AsyncMock(return_value="子问题B 的假设答案"),
            ),
            patch("core.graph.generate_sub_answers", new=AsyncMock(return_value=[("子问题A", "A答案"), ("子问题B", "B答案")])),
            patch("core.graph.synthesize_final_answer", new=AsyncMock(return_value="综合回答")) as synthesis,
        ):
            final_state = await graph.ainvoke(_initial_state(max_retries=1))

        self.assertEqual(final_state["answer"], "综合回答")
        self.assertEqual(final_state["retry_count"], 1)
        self.assertEqual(final_state["sub_queries"], ["子问题A", "子问题B"])
        self.assertEqual(final_state["retrieval_decision"]["action"], "accept")
        synthesis.assert_awaited_once()


class StreamQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_is_idempotent(self):
        from core.stream_queue import StreamQueueRegistry

        registry = StreamQueueRegistry()
        queue_id, queue = registry.create()

        await registry.close(queue_id)
        await registry.close(queue_id)

        self.assertIsNone(await queue.get())
        self.assertTrue(queue.empty())


class RAGServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_anonymous_sessions_do_not_share_default_thread(self):
        from services.rag_service import RAGService

        first = RAGService.resolve_session_id(None, None)
        second = RAGService.resolve_session_id(None, None)

        self.assertNotEqual(first, "default")
        self.assertNotEqual(first, second)

    async def test_stream_graph_error_emits_error_and_done(self):
        from services.rag_service import RAGService

        class FailingGraph:
            async def astream(self, *args, **kwargs):
                raise RuntimeError("sensitive internal detail")
                yield

        service = RAGService(vector_store=object(), rag_graph=FailingGraph())

        async def consume():
            return b"".join(
                [chunk async for chunk in service.ask_stream("问题", session_id="session-1")]
            )

        payload = await asyncio.wait_for(consume(), timeout=1)

        self.assertIn(b"event: error", payload)
        self.assertIn(b"event: done", payload)
        self.assertNotIn(b"sensitive internal detail", payload)

    async def test_same_session_reuses_checkpointed_conversation_history(self):
        from core.graph import build_rag_graph
        from services.rag_service import RAGService

        histories = []

        async def fake_chitchat_generate(**kwargs):
            histories.append(kwargs.get("conversation_history", []))
            yield "闲聊回答"

        service = RAGService(
            vector_store=object(),
            rag_graph=build_rag_graph(object(), FakeReranker()),
        )
        with (
            patch("core.graph.classify_intent_async", new=AsyncMock(return_value="greeting")),
            patch("core.nodes.chitchat.llm_generate_stream", new=fake_chitchat_generate),
        ):
            await service.ask("第一轮", session_id="same-session")
            await service.ask("第二轮", session_id="same-session")

        self.assertEqual(histories[0], [])
        self.assertEqual(
            histories[1],
            [
                {"role": "user", "content": "第一轮"},
                {"role": "assistant", "content": "闲聊回答"},
            ],
        )


class APISessionTests(unittest.TestCase):
    def test_stream_response_exposes_resolved_session_header(self):
        from fastapi.testclient import TestClient
        from main import AppState, create_app

        class FakeRAGService:
            @staticmethod
            def resolve_session_id(session_id, user_id):
                return session_id or user_id or "generated-session"

            async def ask_stream(self, **kwargs):
                yield b"event: done\ndata: {}\n\n"

        app = create_app(
            app_state=AppState(
                vector_store=object(),
                rag_graph=object(),
                rag_service=FakeRAGService(),
                document_service=object(),
                knowledge_service=object(),
                analytics_service=object(),
            ),
            initialize_runtime=False,
        )

        response = TestClient(app).post(
            "/api/v1/rag/ask-stream",
            json={"query": "问题", "session_id": "session-42"},
        )

        self.assertEqual(response.headers["X-Session-ID"], "session-42")


def _initial_state(*, max_retries: int) -> dict:
    return {
        "query": "什么是浮力",
        "subject": None,
        "grade": None,
        "session_id": "session-1",
        "intent": "",
        "complexity": "",
        "retrieved_docs": [],
        "answer": "",
        "retry_count": 0,
        "max_retries": max_retries,
        "conversation_history": [],
        "retrieval_plan": {"strategy": "initial", "queries": ["什么是浮力"]},
        "retrieval_attempts": [],
        "retrieval_metrics": {},
        "retrieval_decision": {},
        "abstain_reason": "",
        "sub_queries": [],
        "_queue_id": "",
    }


if __name__ == "__main__":
    unittest.main(verbosity=2)
