"""评估结果的数据模型"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalSample:
    """单个样本的评估得分"""
    question: str
    answer: str
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalResult:
    """RAGAS 批量评估的完整结果"""
    metrics: list[str]
    scores: dict[str, float]          # 全局聚合分数
    sample_count: int
    samples: list[EvalSample] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def _json_safe_metric_value(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def sanitize_for_json_storage(obj: Any) -> Any:
    """递归将 NaN/inf 转为 None，保证可 JSON 序列化（入库、导出）。"""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_for_json_storage(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json_storage(v) for v in obj]
    return obj


def eval_result_to_dict(er: EvalResult) -> dict:
    """将 EvalResult 转为可序列化的 dict"""
    return {
        "metrics": er.metrics,
        "scores": {k: _json_safe_metric_value(v) for k, v in er.scores.items()},
        "sample_count": er.sample_count,
        "samples": [
            {
                "question": s.question[:200],
                "answer": s.answer[:200],
                "scores": {k: _json_safe_metric_value(v) for k, v in s.scores.items()},
            }
            for s in er.samples
        ],
        **er.extra,
    }
