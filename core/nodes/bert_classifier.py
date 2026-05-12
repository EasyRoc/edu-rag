"""第二层意图识别：BERT 轻量分类器（速度 5-50ms，仅在第一层未命中时执行）"""

import time
from config import settings
from utils.logger import logger

INTENT_CATEGORIES = [
    "educational",
    "chitchat",
    "technical",
    "command",
    "greeting",
    "other",
]

# 延迟加载的 BERT pipeline（模块级单例）
_bert_pipeline = None
_bert_available = None  # None=未探测, True/False=已探测


def _load_bert():
    """加载 BERT 分类模型（首次调用时延迟加载，避免拖慢启动）"""
    global _bert_pipeline, _bert_available

    if _bert_available is not None:
        return _bert_pipeline

    try:
        from transformers import pipeline

        logger.info("正在加载 BERT 意图分类模型 (bert-base-chinese)...")
        _bert_pipeline = pipeline(
            "zero-shot-classification",
            model="bert-base-chinese",
            device=-1,  # CPU
        )
        _bert_available = True
        logger.info("BERT 模型加载成功")
    except Exception as e:
        _bert_available = False
        _bert_pipeline = None
        logger.warning(f"BERT 模型加载失败，将降级到 keyword + LLM 模式: {e}")

    return _bert_pipeline


def bert_classify(query: str) -> dict | None:
    """
    第二层：BERT 零样本分类。

    返回:
        {"intent": str, "confidence": float, "source": "bert", "processing_time_ms": float}
        或 None（BERT 不可用或置信度不足）
    """
    pipeline = _load_bert()
    if pipeline is None:
        return None

    start = time.perf_counter()

    try:
        result = pipeline(
            query,
            candidate_labels=INTENT_CATEGORIES,
            hypothesis_template="这是一条{}的查询。",
        )
        intent = result["labels"][0]
        confidence = result["scores"][0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"BERT 分类: intent={intent}, confidence={confidence:.3f}, time={elapsed_ms:.1f}ms"
        )

        if confidence >= settings.CONFIDENCE_THRESHOLD:
            return {
                "intent": intent,
                "confidence": round(confidence, 4),
                "source": "bert",
                "processing_time_ms": round(elapsed_ms, 2),
            }

        logger.info(f"BERT 置信度不足 ({confidence:.3f} < {settings.CONFIDENCE_THRESHOLD})，降级到 LLM")
        return None

    except Exception as e:
        logger.error(f"BERT 推理异常: {e}")
        return None
