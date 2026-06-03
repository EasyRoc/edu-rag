"""LLM 意图分类器：关键词未命中时的兜底分类。"""

import json
import re
import time

from langchain_core.messages import HumanMessage

from config import settings
from core.llm import get_chat_model
from utils.logger import logger

CLASSIFY_PROMPT = """将以下用户查询分类到指定的意图类别中。

意图类别：educational（教育学习）、chitchat（闲聊）、technical（技术问题）、command（系统命令）、greeting（问候）、other（其他）

用户查询："{query}"

只返回 JSON 格式：{{"intent": "类别名", "confidence": 0.0~1.0}}，不要其他内容。"""


async def llm_classify(query: str) -> dict:
    """
    第三层：LLM 分类（兜底）。

    返回:
        {"intent": str, "confidence": float, "source": "llm", "processing_time_ms": float}
        失败时返回 {"intent": "other", "confidence": 0.0, ...}
    """
    start = time.perf_counter()

    if not settings.LLM_API_KEY or not settings.ENABLE_LLM_FALLBACK:
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "intent": "other",
            "confidence": 0.0,
            "source": "llm",
            "processing_time_ms": round(elapsed, 2),
        }

    try:
        llm = get_chat_model(
            temperature=0.0,
            max_tokens=128,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        response = await llm.ainvoke([
            HumanMessage(content=CLASSIFY_PROMPT.format(query=query)),
        ])
        raw = response.content.strip()
        elapsed_ms = (time.perf_counter() - start) * 1000

        result = _parse_llm_response(raw)
        logger.info(
            f"LLM 分类: intent={result['intent']}, "
            f"confidence={result['confidence']:.3f}, time={elapsed_ms:.1f}ms"
        )
        return {
            "intent": result["intent"],
            "confidence": result["confidence"],
            "source": "llm",
            "processing_time_ms": round(elapsed_ms, 2),
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(f"LLM 分类超时或异常，返回 other: {e}")
        return {
            "intent": "other",
            "confidence": 0.0,
            "source": "llm",
            "processing_time_ms": round(elapsed_ms, 2),
        }


def _parse_llm_response(raw: str) -> dict:
    """解析 LLM 返回的分类结果"""
    try:
        data = json.loads(raw)
        intent = data.get("intent", "other")
        confidence = float(data.get("confidence", 0.5))
        return {"intent": intent, "confidence": confidence}
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^}]+\}', raw)
    if match:
        try:
            data = json.loads(match.group())
            return {
                "intent": data.get("intent", "other"),
                "confidence": float(data.get("confidence", 0.5)),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    for intent in ["educational", "chitchat", "technical", "command", "greeting"]:
        if intent in raw.lower():
            return {"intent": intent, "confidence": 0.6}

    logger.warning(f"无法解析 LLM 分类结果，返回 other: {raw[:100]}")
    return {"intent": "other", "confidence": 0.3}
