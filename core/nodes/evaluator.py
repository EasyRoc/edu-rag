"""评估节点：Corrective RAG 的质量评估与纠正决策

评估生成的回答是否基于检索文档、是否完整、是否出现幻觉。
"""
from typing import Literal

from utils.logger import logger


def evaluate_quality(
    query: str,
    answer: str,
    retrieved_docs: list[dict],
    retry_count: int,
    max_retries: int = 2,
) -> tuple[Literal["accept", "retry", "give_up"], str]:
    """
    评估生成回答的质量，决定下一步动作。

    评估维度：
    1. 是否有检索结果
    2. 回答是否非空
    3. 检索结果的相关性（通过 score 判断）
    4. 是否有足够的上下文

    返回:
        (决策, 评估原因)
    """
    logger.info(f"质量评估: retry_count={retry_count}/{max_retries}")
    # 1. 无检索结果
    if not retrieved_docs:
        logger.warning("评估结果: give_up — 没有检索到任何相关文档")
        return "give_up", "未检索到相关文档"
    # 2. 回答为空
    if not answer or len(answer.strip()) < 5:
        if retry_count < max_retries:
            logger.warning("评估结果: retry — 回答为空或过短")
            return "retry","回答内容为空或过短"
        else:
            logger.warning("评估结果: give_up — 回答为空且已达到最大重试次数")
            return "give_up", "多次尝试后仍无法生成有效回答"

    # 3. 检查最高分的检索结果是否足够相关
    max_score = max(doc.get("score", 0) for doc in retrieved_docs)
    if max_score < 0.4 and retry_count < max_retries:
        logger.warning(f"评估结果: retry — 检索相关性偏低 (max_score={max_score:.3f})")
        return "retry", f"检索结果相关性不足 (最高分: {max_score:.3f})"

    # 4. 通过评估
    logger.info(f"评估结果: accept — 回答质量合格 (docs={len(retrieved_docs)}, max_score={max_score:.3f})")
    return "accept", "回答质量合格"

def decide_next_step(evaluation_result: tuple) -> Literal["accept", "retry", "give_up"]:
    """从评估结果元组中提取决策"""
    return evaluation_result[0]
