"""评估节点：Corrective RAG 的检索质量评估与纠正决策

评估检索结果是否足以支撑回答，决定接受/重试/放弃。
"""
from typing import Literal

from config import settings
from utils.logger import logger


def evaluate_retrieval(
    retrieved_docs: list[dict],
    retry_count: int = 0,
    max_retries: int = 2,
) -> tuple[Literal["accept", "retry", "give_up"], str]:
    """
    评估检索结果质量，决定下一步动作。

    评估维度（按优先级）：
    1. 检索结果是否为空
    2. 结果数量和平均相关性是否达标
    3. top-1 score 是否过低

    返回:
        (决策, 评估原因)
    """
    threshold = getattr(settings, "RETRIEVAL_QUALITY_THRESHOLD", 0.5)
    min_docs = getattr(settings, "RETRIEVAL_MIN_DOCS", 2)

    if not retrieved_docs:
        if retry_count < max_retries:
            logger.warning("评估结果: retry — 未检索到任何文档")
            return "retry", "未检索到任何文档"
        logger.warning("评估结果: give_up — 多次检索均无结果")
        return "give_up", "多次检索均未找到相关文档"

    avg_score = sum(d["score"] for d in retrieved_docs) / len(retrieved_docs)
    top_score = retrieved_docs[0]["score"]
    enough_docs = len(retrieved_docs) >= min_docs

    if avg_score >= threshold and enough_docs:
        logger.info(f"评估结果: accept — avg={avg_score:.3f}, top1={top_score:.3f}, count={len(retrieved_docs)}")
        return "accept", "检索质量合格"

    if retry_count < max_retries:
        reason_parts = []
        if avg_score < threshold:
            reason_parts.append(f"avg_score={avg_score:.3f} < {threshold}")
        if not enough_docs:
            reason_parts.append(f"count={len(retrieved_docs)} < {min_docs}")
        reason = "检索质量不足: " + ", ".join(reason_parts)
        logger.warning(f"评估结果: retry — {reason}")
        return "retry", reason

    logger.warning(f"评估结果: give_up — 重试{max_retries}次后质量仍不达标 (avg={avg_score:.3f})")
    return "give_up", f"多次重试后检索质量仍不达标 (avg_score={avg_score:.3f})"
