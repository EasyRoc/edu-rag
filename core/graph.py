from typing import TypedDict, Literal

from core.nodes.chitchat import chitchat_node
from core.nodes.evaluator import evaluate_quality
from core.nodes.generator import llm_generate_stream
from core.nodes.query_classifier import classify_query, classify_intent_async
from core.nodes.retriever import hybrid_retrieve
from core.stream_queue import _registry as _stream_queues
from core.vectorestore import K12VectorStore
from utils.logger import logger
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

MAX_ROUNDS = 10  # 最多保留的对话轮数

# ==================== 状态定义 ====================
class RAGState(TypedDict):
    """RAG 流程的全局状态"""
    query: str  # 用户原始查询
    subject: str | None  # 学科过滤
    grade: str | None  # 年级过滤
    intent: str  # 查询意图: educational / chitchat
    complexity: str  # 查询复杂度: simple / medium / complex
    retrieved_docs: list  # 检索结果文档列表
    answer: str  # 生成的回答
    evaluation_reason: str  # 评估结果原因
    evaluation_decision: str  # 评估决策: accept / retry / give_up
    retry_count: int  # 当前重试次数
    max_retries: int  # 最大重试次数
    conversation_history: list[dict]  # 短期记忆: [{"role": "user"/"assistant", "content": "..."}]
    # 运行时注入（不在 TypedDict 中声明会导致 astream 过滤掉这些 key）
    _queue_id: str  # 流式 token 队列 ID，通过全局 _stream_queues 查找


# ==================== 图节点函数 ====================
async def finalize_node(state: RAGState) -> dict:
    """记忆持久化节点：将当前 Q&A 追加到 conversation_history 并裁剪"""
    history = list(state.get("conversation_history", []))
    history.append({"role": "user", "content": state["query"]})
    history.append({"role": "assistant", "content": state.get("answer", "")})
    max_msgs = MAX_ROUNDS * 2
    if len(history) > max_msgs:
        history = history[-max_msgs:]
    return {"conversation_history": history}


async def classify_node(state: RAGState) -> dict:
    """查询分类节点：三层意图识别 + 教育类查询复杂度分级"""
    logger.info(f"[节点] classify: query='{state['query'][:50]}'")
    intent = await classify_intent_async(state["query"])
    if intent != "educational":
        logger.info(f"[节点] classify: 非教育类意图 ({intent})，跳过复杂度分级")
        return {"intent": intent, "complexity": "simple"}
    complexity = classify_query(state["query"])
    return {"intent": intent, "complexity": complexity}


async def retrieve_node(state: RAGState) -> dict:
    """混合检索节点（策略驱动）"""
    logger.info(f"[节点] retrieve: complexity={state['complexity']}")
    vector_store: K12VectorStore = state.get("_vector_store")
    docs = await hybrid_retrieve(
        vector_store=vector_store,
        query=state["query"],
        complexity=state["complexity"],
        intent=state.get("intent", "educational"),
        subject=state.get("subject"),
        grade=state.get("grade"),
    )
    return {"retrieved_docs": docs}


async def generate_node(state: RAGState) -> dict:
    """LLM 生成节点：调用 LLM 生成回答，同时通过全局队列推送 token 实现流式"""
    logger.info(f"[节点] generate: retry_count={state.get('retry_count', 0)}")
    queue_id = state.get("_queue_id")
    stream_queue = _stream_queues.get(queue_id) if queue_id else None
    full_answer = ""
    try:
        async for token in llm_generate_stream(
            query=state["query"],
            context_docs=state.get("retrieved_docs", []),
            conversation_history=state.get("conversation_history", []),
        ):
            full_answer += token
            if stream_queue is not None:
                await stream_queue.put(token)
    finally:
        if stream_queue is not None:
            await stream_queue.put(None)

    logger.info(f"[节点] generate: 完成，共 {len(full_answer)} 字")
    return {"answer": full_answer}


async def evaluate_node(state: RAGState) -> dict:
    """质量评估节点"""
    logger.info(f"[节点] evaluate: retry_count={state.get('retry_count', 0)}")
    decision, reason = evaluate_quality(
        query=state["query"],
        answer=state.get("answer", ""),
        retrieved_docs=state.get("retrieved_docs", []),
        retry_count=state.get("retry_count", 0),
        max_retries=state.get("max_retries", 2),
    )
    return {"evaluation_reason": reason, "evaluation_decision": decision}


async def re_retrieve_node(state: RAGState) -> dict:
    """重新检索节点（纠正时触发，直接扩大检索深度，不触发额外策略）"""
    logger.info(f"[节点] re_retrieve: 第 {state.get('retry_count', 0) + 1} 次重试")
    vector_store: K12VectorStore = state.get("_vector_store")
    docs = vector_store.hybrid_search(
        query=state["query"],
        subject=state.get("subject"),
        grade=state.get("grade"),
        top_k=8,  # 重试时扩大检索
    )
    return {
        "retrieved_docs": docs,
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ==================== 图构建 ====================
def should_continue(state: RAGState) -> Literal["accept", "retry", "give_up"]:
    """条件边：根据评估结果决定流程走向"""
    decision = state.get("evaluation_decision", "give_up")
    if decision == "accept":
        logger.info("[条件边] 评估通过 -> 结束")
        return "accept"
    elif decision == "retry" and state.get("retry_count", 0) < state.get("max_retries", 2):
        logger.info(f"[条件边] 需要重试 (第 {state.get('retry_count', 0) + 1} 次)")
        return "retry"
    else:
        logger.info("[条件边] 无法完成 -> 结束")
        return "give_up"


def build_rag_graph(vector_store: K12VectorStore):
    """
    构建 RAG 流程的 LangGraph。

    Args:
        vector_store: 向量存储实例

    Returns:
        编译后的 LangGraph 应用
    """
    logger.info("正在构建 LangGraph RAG 工作流...")

    # 注入 vector_store（通过闭包方式绑定）
    async def retrieve_with_store(state: RAGState) -> dict:
        state["_vector_store"] = vector_store
        return await retrieve_node(state)

    async def re_retrieve_with_store(state: RAGState) -> dict:
        state["_vector_store"] = vector_store
        return await re_retrieve_node(state)

    # 构建图
    workflow = StateGraph(RAGState)

    # 添加节点
    workflow.add_node("classify", classify_node)
    workflow.add_node("retrieve", retrieve_with_store)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("re_retrieve", re_retrieve_with_store)
    workflow.add_node("chitchat", chitchat_node)
    workflow.add_node("finalize", finalize_node)

    # 设置入口
    workflow.set_entry_point("classify")

    # 条件边：根据意图分流
    #   - educational → 继续 RAG 管线（检索 → 生成 → 评估）
    #   - 其它 → 闲聊回复
    def route_by_intent(state: RAGState) -> Literal["retrieve", "chitchat"]:
        return "retrieve" if state.get("intent") == "educational" else "chitchat"

    workflow.add_conditional_edges("classify", route_by_intent, {
        "retrieve": "retrieve",
        "chitchat": "chitchat",
    })
    workflow.add_edge("chitchat", "finalize")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate")

    # 条件边：从 evaluate 根据质量分流向
    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "accept": "finalize",
            "retry": "re_retrieve",
            "give_up": "finalize",
        },
    )
    workflow.add_edge("re_retrieve", "generate")
    workflow.add_edge("finalize", END)

    app = workflow.compile(checkpointer=MemorySaver())
    logger.info("LangGraph RAG 工作流构建完成")
    return app
