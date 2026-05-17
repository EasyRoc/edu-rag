"""多策略检索模块

策略选择器根据意图和复杂度自动决定检索策略：
- DIRECT: 直接混合检索（simple 查询）
- MULTI_QUERY: 多查询变体 + RRF 融合（medium 查询）
- DECOMPOSITION: 复杂问题拆解（complex 查询）

补充策略在首轮检索质量不足时自动触发：
- Step-Back: 抽象回退问题检索背景知识
- HyDE: 假设答案 embedding 检索
"""
from core.strategies.selector import (
    StrategyType,
    select_strategy,
    assess_retrieval_quality,
    should_apply_hyde,
    should_apply_step_back,
)
from core.strategies.multi_query import generate_query_variants, multi_query_fusion
from core.strategies.decomposition import decompose_query, merge_sub_results
from core.strategies.step_back import generate_step_back_query
from core.strategies.hyde import generate_hypothetical_answer

__all__ = [
    "StrategyType",
    "select_strategy",
    "assess_retrieval_quality",
    "should_apply_hyde",
    "should_apply_step_back",
    "generate_query_variants",
    "multi_query_fusion",
    "decompose_query",
    "merge_sub_results",
    "generate_step_back_query",
    "generate_hypothetical_answer",
]
