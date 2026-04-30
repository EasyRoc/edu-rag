from typing import TypedDict, Literal

from core.nodes.evaluator import evaluate_quality, decide_next_step
from core.nodes.generator import llm_generate
from core.nodes.query_classifier import classify_query
from core.nodes.retriever import hybrid_retrieve
from core.vectorestore import K12VectorStore
from utils.logger import logger
from langgraph.graph import StateGraph, END


# ==================== 状态定义 ====================
class RAGState(TypedDict):
    """RAG 流程的全局状态"""
    query: str  # 用户原始查询
    subject: str | None  # 学科过滤
    grade: str | None  # 年级过滤
    complexity: str  # 查询复杂度: simple / medium / complex
    retrieved_docs: list  # 检索结果文档列表
    answer: str  # 生成的回答
    evaluation_reason: str  # 评估结果原因
    retry_count: int  # 当前重试次数
    max_retries: int  # 最大重试次数


# ==================== 图节点函数 ====================
async def classify_node(state: RAGState) -> dict:
    """查询分类节点"""
    logger.info(f"[节点] classify: query='{state['query'][:50]}'")
    complexity = classify_query(state["query"])
    return {"complexity": complexity}


async def retrieve_node(state: RAGState) -> dict:
    """混合检索节点"""
    logger.info(f"[节点] retrieve: complexity={state['complexity']}")
    vector_store: K12VectorStore = state.get("_vector_store")
    docs = hybrid_retrieve(
        vector_store=vector_store,
        query=state["query"],
        complexity=state["complexity"],
        subject=state.get("subject"),
        grade=state.get("grade"),
    )
    return {"retrieved_docs": docs}


async def generate_node(state: RAGState) -> dict:
    """LLM 生成节点"""
    logger.info(f"[节点] generate: retry_count={state.get('retry_count', 0)}")
    answer = await llm_generate(
        query=state["query"],
        context_docs=state.get("retrieved_docs", []),
    )
    return {"answer": answer}


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
    return {"evaluation_reason": reason}


async def re_retrieve_node(state: RAGState) -> dict:
    """重新检索节点（纠正时触发，增加检索深度）"""
    logger.info(f"[节点] re_retrieve: 第 {state.get('retry_count', 0) + 1} 次重试")
    vector_store: K12VectorStore = state.get("_vector_store")
    # 增加 top_k 获取更多结果
    docs = hybrid_retrieve(
        vector_store=vector_store,
        query=state["query"],
        complexity="complex",  # 重试时使用 complex 深度
        subject=state.get("subject"),
        grade=state.get("grade"),
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

    # 设置入口
    workflow.set_entry_point("classify")

    # 连接边
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate")

    # 条件边：从 evaluate 根据质量分流向
    workflow.add_conditional_edges(
        "evaluate",
        should_continue,
        {
            "accept": END,
            "retry": "re_retrieve",
            "give_up": END,
        },
    )
    workflow.add_edge("re_retrieve", "generate")
    app = workflow.compile()
    logger.info("LangGraph RAG 工作流构建完成")
    return app
