"""策略选择器：根据意图、复杂度和查询特征决定检索策略"""
from enum import Enum
from config import settings
from utils.logger import logger


class StrategyType(Enum):
    DIRECT = "direct"              # 直接检索
    MULTI_QUERY = "multi_query"    # 多查询变体 + RRF 融合
    DECOMPOSITION = "decomposition"  # 复杂问题拆解


def select_strategy(intent: str, complexity: str, query: str = "") -> StrategyType:
    """根据意图和复杂度选择主检索策略"""
    # 非教育类一律直接检索
    if intent != "educational":
        return StrategyType.DIRECT

    # simple 查询直接检索
    if complexity == "simple":
        return StrategyType.DIRECT

    # medium 查询使用多查询策略
    if complexity == "medium":
        return StrategyType.MULTI_QUERY

    # complex 查询使用分解策略
    if complexity == "complex":
        return StrategyType.DECOMPOSITION

    return StrategyType.DIRECT


def assess_retrieval_quality(docs: list[dict]) -> bool:
    """评估检索结果质量是否达标"""
    threshold = getattr(settings, 'RETRIEVAL_QUALITY_THRESHOLD', 0.5)
    min_docs = getattr(settings, 'STEP_BACK_MIN_DOCS', 3)

    if not docs:
        logger.info("检索质量评估: 无结果，不达标")
        return False

    # 平均分达标且结果数足够
    avg_score = sum(d["score"] for d in docs) / len(docs)
    top1_score = docs[0]["score"]
    has_enough = len(docs) >= min_docs

    passed = (avg_score >= threshold) and has_enough
    logger.info(
        f"检索质量评估: avg_score={avg_score:.3f}, top1={top1_score:.3f}, "
        f"count={len(docs)}, passed={passed}"
    )
    return passed


def should_apply_hyde(docs: list[dict]) -> bool:
    """判断是否需要 HyDE 补充策略（首轮检索置信度很低时）"""
    hyde_min = getattr(settings, 'HYDE_MIN_SCORE', 0.4)
    if not docs:
        return True
    top1 = docs[0]["score"]
    return top1 < hyde_min


def should_apply_step_back(docs: list[dict]) -> bool:
    """判断是否需要 Step-Back 补充策略（结果少或平均分低）"""
    threshold = getattr(settings, 'RETRIEVAL_QUALITY_THRESHOLD', 0.5)
    min_docs = getattr(settings, 'STEP_BACK_MIN_DOCS', 3)
    if not docs:
        return True
    avg_score = sum(d["score"] for d in docs) / len(docs)
    return len(docs) < min_docs or avg_score < threshold
