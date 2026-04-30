"""Adaptive RAG 查询分类节点：根据查询复杂度进行分级路由

根据查询长度、关键词、学科特点等将查询分为 simple / medium / complex 三级。
"""
from typing import Literal

from utils.logger import logger

# 简单查询关键词：以事实性、定义类问题为主
_SIMPLE_KEYWORDS = [
    "是什么", "什么是", "定义", "公式", "定理", "等于", "多少",
    "谁", "哪一年", "在哪里", "什么时候",
]

# 复杂查询关键词：需要比较、分析、综合
_COMPLEX_KEYWORDS = [
    "比较", "对比", "区别", "异同", "关系", "分析", "为什么",
    "如何影响", "原理", "推导", "证明", "总结",
]

def classify_query(query: str) -> Literal["simple", "medium", "complex"]:
    """
    基于规则对查询进行分类。

    规则：
    - simple: 事实性、定义类查询，长度短且包含简单关键词
    - complex: 需要比较、分析、推理的查询
    - medium: 介于两者之间
    """

    query_lower = query.strip().lower()
    query_len = len(query_lower)
    logger.info(f"查询分类: query='{query[:50]}', 长度={query_len}")
    # 复杂查询检测
    has_complex = any(kw in query_lower for kw in _COMPLEX_KEYWORDS)
    if has_complex and query_len > 15:
        logger.info(f"分类结果: complex (包含分析/比较关键词)")
        return "complex"

    # 简单查询检测
    has_simple = any(kw in query_lower for kw in _SIMPLE_KEYWORDS)
    if has_simple or query_len < 10:
        logger.info(f"分类结果: simple (简单定义/事实查询)")
        return "simple"

    # 默认中等复杂度
    logger.info(f"分类结果: medium")
    return "medium"


