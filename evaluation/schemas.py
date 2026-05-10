"""评估结果的数据模型"""

from __future__ import annotations

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


def eval_result_to_dict(er: EvalResult) -> dict:
    """将 EvalResult 转为可序列化的 dict"""
    return {
        "metrics": er.metrics,
        "scores": er.scores,
        "sample_count": er.sample_count,
        "samples": [
            {
                "question": s.question[:200],
                "answer": s.answer[:200],
                "scores": s.scores,
            }
            for s in er.samples
        ],
        **er.extra,
    }
