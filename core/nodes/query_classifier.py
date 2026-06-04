"""自适应意图识别 + 复杂度分级

两层渐进式意图分类（按速度优先）：
1. 关键词匹配 (< 1ms) — 命中则短路返回
2. LLM 兜底 (200-800ms) — 关键词未命中时执行，结果自动收集为后续分类器训练数据

复杂度分级同样采用规则 + LLM 兜底：
1. 规则层 — 关键词命中且无冲突时直接返回
2. LLM 层 — 规则层返回 medium 或关键词冲突时兜底
"""

import asyncio
import time
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from core.llm import get_chat_model
from core.nodes.keyword_matcher import match_keywords
from core.nodes.llm_classifier import llm_classify
from core.nodes.training_collector import save_case
from utils.logger import logger

# ==================== 复杂度分级 — 规则层 ====================
_SIMPLE_KEYWORDS = [
    "是什么", "什么是", "定义", "公式", "定理", "等于", "多少",
    "谁", "哪一年", "什么时候", "在哪里",
    "何时", "何地", "何人",
    "列出", "列举", "写出",
    "简称", "缩写", "全称",
    "怎么读", "怎么念", "怎么发音",
]

_COMPLEX_KEYWORDS = [
    "比较", "对比", "区别", "异同", "关系", "分析", "为什么",
    "如何影响", "原理", "推导", "证明", "总结",
    "归纳", "阐述", "评价", "论证", "评述",
    "成因", "机制", "影响因素", "关联",
    "优缺点", "利弊", "优劣",
    "原因", "导致", "造成",
    "联系", "结合", "综合",
]


def classify_query(query: str) -> Literal["simple", "medium", "complex"]:
    """规则层复杂度分类（同步，<1ms）。

    - simple/complex 关键词命中且无冲突 → 直接返回
    - 关键词冲突（同时命中 simple 和 complex）→ 返回 medium 交由 LLM 兜底
    - 无关键词命中 → medium
    """
    query_lower = query.strip().lower()

    has_simple = any(kw in query_lower for kw in _SIMPLE_KEYWORDS)
    has_complex = any(kw in query_lower for kw in _COMPLEX_KEYWORDS)

    # 冲突：同时命中 simple 和 complex → 交给 LLM 判断
    if has_simple and has_complex:
        logger.info("复杂度规则: simple + complex 关键词冲突 → medium (交 LLM)")
        return "medium"

    if has_complex:
        logger.info("复杂度规则: complex")
        return "complex"

    if has_simple:
        logger.info("复杂度规则: simple")
        return "simple"

    logger.info("复杂度规则: 无匹配 → medium")
    return "medium"


# ==================== 复杂度分级 — LLM 兜底层 ====================
class ComplexityResult(BaseModel):
    complexity: Literal["simple", "medium", "complex"] = Field(
        description="查询复杂度级别"
    )
    reason: str = Field(description="分类理由，一句话")


COMPLEXITY_PROMPT = """分析以下教育类查询的复杂度，分为三级：

- simple: 简单事实检索，答案直接、唯一（如定义、公式、日期、人名、地点）
- medium: 需要解释概念、举例说明、描述过程或简要概括
- complex: 涉及比较分析、多概念综合、因果推理、推导证明或需要深度阐述

用户查询："{query}"

只输出 JSON。"""


async def classify_query_llm(query: str) -> ComplexityResult:
    """LLM 兜底复杂度分类，带结构化输出（含理由）。"""
    llm = get_chat_model(temperature=0.0, max_tokens=256, timeout=10.0)
    structured = llm.with_structured_output(ComplexityResult, method="json_mode")
    return await structured.ainvoke([
        HumanMessage(content=COMPLEXITY_PROMPT.format(query=query)),
    ])


async def classify_query_with_fallback(query: str) -> Literal["simple", "medium", "complex"]:
    """复杂度分类完整管线：规则层 → LLM 兜底。

    规则层命中 simple/complex 直接返回；medium 或冲突走 LLM。
    """
    # 第一层：规则
    rule_result = classify_query(query)
    if rule_result != "medium":
        return rule_result

    # 第二层：LLM 兜底
    try:
        result = await classify_query_llm(query)
        logger.info(
            f"复杂度 LLM 兜底: {result.complexity}, reason={result.reason}"
        )
        return result.complexity
    except Exception as e:
        logger.warning(f"复杂度 LLM 分类失败，回退 medium: {e}")
        return "medium"


# ==================== 意图分类（两层） ====================
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
