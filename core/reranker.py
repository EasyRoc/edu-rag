"""本地 CrossEncoder 重排器。

重排模型较重，因此采用懒加载：应用启动时只创建包装器，第一次真实问答时
再加载模型权重。CrossEncoder 原始输出按 logit 处理，再由本模块统一 sigmoid。
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from typing import Any

from config import settings
from utils.logger import logger


class RerankerUnavailableError(RuntimeError):
    """重排器无法给候选文档打分时抛出。"""


def _sigmoid(value: float) -> float:
    """将 CrossEncoder logit 映射到 0~1，作为在线门控唯一质量分数。"""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class CrossEncoderReranker:
    """按需加载 sentence-transformers CrossEncoder 的轻量包装器。"""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        model_factory: Callable[..., Any] | None = None,
    ):
        self.enabled = settings.ENABLE_RERANKER if enabled is None else enabled
        self.model_name = model_name or settings.RERANKER_MODEL
        self.device = device or settings.RERANKER_DEVICE
        self.batch_size = batch_size or settings.RERANKER_BATCH_SIZE
        self._model_factory = model_factory
        self._model = None

    def _get_model(self):
        if not self.enabled:
            raise RerankerUnavailableError("reranker is disabled")
        if self._model is None:
            factory = self._model_factory
            if factory is None:
                from sentence_transformers import CrossEncoder

                factory = CrossEncoder
            try:
                logger.info("正在加载本地重排模型: model=%s, device=%s", self.model_name, self.device)
                self._model = factory(self.model_name, device=self.device)
                logger.info("本地重排模型加载完成: %s", self.model_name)
            except Exception as exc:
                logger.warning("本地重排模型加载失败: %s", exc)
                raise RerankerUnavailableError("reranker model could not be loaded") from exc
        return self._model

    def _rerank_sync(self, query: str, docs: list[dict]) -> list[dict]:
        """同步执行重排；外层 async 方法会把它放到线程池中运行。"""
        if not docs:
            return []
        model = self._get_model()
        pairs = [(query, str(doc.get("text", ""))) for doc in docs]
        try:
            raw_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                activation_fn=lambda score: score,
                convert_to_numpy=True,
            )
        except Exception as exc:
            logger.warning("本地重排推理失败: %s", exc)
            raise RerankerUnavailableError("reranker inference failed") from exc

        ranked = []
        for doc, raw_score in zip(docs, raw_scores):
            item = dict(doc)
            item["rerank_raw_score"] = float(raw_score)
            item["rerank_score"] = _sigmoid(float(raw_score))
            ranked.append(item)
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        logger.info(
            "重排完成: candidates=%d, top1=%.4f, batch_size=%d",
            len(ranked),
            ranked[0]["rerank_score"] if ranked else 0.0,
            self.batch_size,
        )
        return ranked

    async def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """异步重排入口，避免 CrossEncoder 推理阻塞事件循环。"""
        return await asyncio.to_thread(self._rerank_sync, query, docs)
