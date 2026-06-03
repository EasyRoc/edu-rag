"""策略选择器：根据意图、复杂度和查询特征决定检索策略"""
from enum import Enum


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
