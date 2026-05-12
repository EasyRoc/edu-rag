"""训练数据收集器：LLM 分类结果自动保存为 BERT 微调数据"""

import json
import os
import threading

from utils.logger import logger

# 训练数据文件路径
_TRAINING_FILE = os.path.join(os.path.dirname(__file__), "../../data/intent_training_data.jsonl")
_lock = threading.Lock()


def save_case(query: str, intent: str, confidence: float, source: str, processing_time_ms: float):
    """
    保存一条分类 case 到训练数据文件。

    仅保存 LLM 来源的 case（置信度高、质量好），
    后续用于微调 BERT 分类器。
    """
    if source != "llm":
        return  # 关键词命中的不需要（太简单，BERT 不需要学）

    record = {
        "query": query,
        "intent": intent,
        "confidence": confidence,
        "source": source,
        "processing_time_ms": processing_time_ms,
    }

    try:
        os.makedirs(os.path.dirname(_TRAINING_FILE), exist_ok=True)
        with _lock:
            with open(_TRAINING_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug(f"训练数据已保存: intent={intent}, query={query[:50]}")
    except Exception as e:
        logger.warning(f"保存训练数据失败: {e}")
