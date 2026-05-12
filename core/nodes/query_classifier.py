"""自适应意图识别 + 复杂度分级

两层渐进式意图分类（按速度优先）：
1. 关键词匹配 (< 1ms) — 命中则短路返回
2. LLM 兜底 (200-800ms) — 关键词未命中时执行，结果自动收集为 BERT 训练数据
"""

import asyncio
import time
from typing import Literal

from core.nodes.keyword_matcher import match_keywords
from core.nodes.llm_classifier import llm_classify
from core.nodes.training_collector import save_case
from utils.logger import logger

# ==================== 复杂度分级（规则，不需要三层） ====================
_SIMPLE_KEYWORDS = [
    "是什么", "什么是", "定义", "公式", "定理", "等于", "多少",
    "谁", "哪一年", "在哪里", "什么时候",
]

_COMPLEX_KEYWORDS = [
    "比较", "对比", "区别", "异同", "关系", "分析", "为什么",
    "如何影响", "原理", "推导", "证明", "总结",
]


def classify_query(query: str) -> Literal["simple", "medium", "complex"]:
    """基于规则对教育类查询进行复杂度分类"""
    query_lower = query.strip().lower()
    query_len = len(query_lower)

    has_complex = any(kw in query_lower for kw in _COMPLEX_KEYWORDS)
    if has_complex and query_len > 15:
        logger.info(f"复杂度结果: complex")
        return "complex"

    has_simple = any(kw in query_lower for kw in _SIMPLE_KEYWORDS)
    if has_simple or query_len < 10:
        logger.info(f"复杂度结果: simple")
        return "simple"

    logger.info(f"复杂度结果: medium")
    return "medium"


# ==================== 意图分类（三层） ====================
# 归类到 educational 的意图（走 RAG 检索管线）
_RAG_INTENTS = {"educational"}
# 归类到闲聊的意图（走 chitchat 节点）
_CHITCHAT_INTENTS = {"chitchat", "greeting", "technical", "command", "other"}


def classify_intent(query: str) -> str:
    """
    两层意图分类主入口（同步）。

    关键词命中直接返回，未命中用 LLM 兜底（通过 asyncio.run 桥接）。
    在 async graph 节点中应使用 classify_intent_async。

    返回意图字符串。
    """
    start = time.perf_counter()

    # —— 第一层：关键词匹配 ——
    result = match_keywords(query)
    if result:
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"意图识别: intent={result['intent']}, source=keyword, "
            f"confidence={result['confidence']}, time={elapsed:.1f}ms"
        )
        return result["intent"]

    # —— 第二层：LLM（需要事件循环） ——
    try:
        asyncio.get_running_loop()
        logger.warning("在运行中的事件循环调用同步 classify_intent，LLM 层不可用，暂返回 other")
        return "other"
    except RuntimeError:
        try:
            result = asyncio.run(llm_classify(query))
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                f"意图识别: intent={result['intent']}, source=llm, "
                f"confidence={result['confidence']}, time={elapsed:.1f}ms"
            )
            save_case(query, result["intent"], result["confidence"],
                      result["source"], result["processing_time_ms"])
            return result["intent"]
        except Exception as e:
            logger.error(f"LLM 分类失败: {e}")
            return "other"


async def classify_intent_async(query: str) -> str:
    """
    两层意图分类主入口（异步版本，用于 graph 节点中）。

    关键词 → LLM，LLM 结果自动存入训练数据。
    """
    start = time.perf_counter()

    # —— 第一层：关键词匹配 ——
    result = match_keywords(query)
    if result:
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"意图识别: intent={result['intent']}, source=keyword, "
            f"confidence={result['confidence']}, time={elapsed:.1f}ms"
        )
        return result["intent"]

    # —— 第二层：LLM ——
    try:
        result = await llm_classify(query)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"意图识别: intent={result['intent']}, source=llm, "
            f"confidence={result['confidence']}, time={elapsed:.1f}ms"
        )
        save_case(query, result["intent"], result["confidence"],
                  result["source"], result["processing_time_ms"])
        return result["intent"]
    except Exception as e:
        logger.error(f"LLM 分类失败: {e}")
        return "other"
