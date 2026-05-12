"""基于 RAGAS 的 RAG 质量评估器

支持指标：
  - faithfulness: 回答是否忠实于检索上下文（需 LLM）
  - answer_relevancy: 回答与问题的相关性（需 LLM + Embedding）
  - context_precision: 检索上下文是否包含无关信息（需 LLM）
  - context_recall: 检索上下文是否覆盖答案所需信息（需 LLM + ground_truth）

用法:
    evaluator = RAGASEvaluator()
    result = await evaluator.evaluate_dataset(dataset)
    print(result.scores)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from datasets import Dataset
from openai import OpenAI

from config import settings
from evaluation.schemas import EvalResult, EvalSample
from utils.logger import logger

logger = logging.getLogger(__name__)


class _LangChainStyleEmbeddingsAdapter:
    """将 RAGAS ``BaseRagasEmbedding``（``embed_text`` / ``embed_texts``）适配为
    旧版 ``AnswerRelevancy`` 所用的 LangChain 风格 ``embed_query`` / ``embed_documents``。
    """

    def __init__(self, inner: Any):
        self._inner = inner

    def embed_query(self, text: str) -> list[float]:
        return list(self._inner.embed_text(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(vec) for vec in self._inner.embed_texts(texts)]


# ---------------------------------------------------------------------------
# RAGAS 指标  —— 延迟导入，避免未安装时炸裂
# ---------------------------------------------------------------------------
def _build_ragas_metrics(
    metric_names: list[str],
    llm: Any,
    embeddings: Any | None = None,
) -> list:
    """根据名称列表创建 RAGAS 0.4.x 指标实例。

    须使用 ``ragas.metrics._*`` 下的类：``ragas.evaluate`` 要求 ``isinstance(m, Metric)``，
    而 ``ragas.metrics.collections.*`` 中的同名类基于 ``SimpleBaseMetric``，不会通过该检查。
    """
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    registry = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
    }

    result = []
    for name in metric_names:
        if name in registry:
            result.append(registry[name]())
        else:
            logger.warning("未知指标: %s，已跳过", name)
    return result


class RAGASEvaluator:
    """封装 RAGAS 评估逻辑，复用项目已有的 LLM / Embedding 配置"""

    def __init__(self):
        # ---------- LLM  ----------
        # RAGAS 0.4.x 需要原生的 openai.Client，通过 llm_factory 包装
        api_key = settings.LLM_API_KEY or "empty"
        base_url = settings.LLM_BASE_URL.rstrip("/")
        self._openai_client = OpenAI(api_key=api_key, base_url=base_url)
        self._ragas_llm = self._build_ragas_llm()

        # ---------- Embedding ----------
        self._ragas_embeddings = self._build_ragas_embeddings()

        logger.info(
            "RAGASEvaluator 就绪  llm=%s  base_url=%s  embedding=%s",
            settings.LLM_MODEL,
            settings.LLM_BASE_URL,
            settings.EMBEDDING_MODEL,
        )

    # ------------------------------------------------------------------
    # LLM / Embedding 工厂
    # ------------------------------------------------------------------
    def _build_ragas_llm(self) -> Any:
        """用项目 LLM 配置构建 RAGAS LLM 包装"""
        from ragas.llms import llm_factory

        if not settings.LLM_API_KEY:
            logger.warning("LLM_API_KEY 未配置，RAGAS 评估无法调用 LLM，结果不可靠")

        logger.info(
            "RAGAS LLM Instructor max_tokens=%d（Faithfulness 等需足够大以防 JSON 截断）",
            settings.RAGAS_LLM_MAX_TOKENS,
        )
        return llm_factory(
            settings.LLM_MODEL,
            client=self._openai_client,
            max_tokens=settings.RAGAS_LLM_MAX_TOKENS,
        )

    def _build_ragas_embeddings(self) -> Any:
        """用项目 Embedding 模型构建 RAGAS Embedding 包装"""
        from ragas.embeddings import HuggingFaceEmbeddings

        logger.info("正在加载 Embedding 模型: %s", settings.EMBEDDING_MODEL)
        inner = HuggingFaceEmbeddings(
            model=settings.EMBEDDING_MODEL,
            device=settings.EMBEDDING_DEVICE,
            normalize_embeddings=True,
        )
        return _LangChainStyleEmbeddingsAdapter(inner)

    # ------------------------------------------------------------------
    # 单样本评估
    # ------------------------------------------------------------------
    async def evaluate_sample(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> EvalSample:
        """评估单个问答对"""
        ds_dict = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            ds_dict["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(ds_dict)
        result = await self._do_evaluate(dataset)
        return result.samples[0]

    # ------------------------------------------------------------------
    # 批量评估
    # ------------------------------------------------------------------
    async def evaluate_dataset(
        self,
        dataset: Dataset,
        metrics: list[str] | None = None,
    ) -> EvalResult:
        """对 HuggingFace Dataset 执行批量 RAGAS 评估

        Dataset 需包含列:
          - question (str)
          - answer (str)
          - contexts (list[str])
          - ground_truth (str, 可选 —— 用于 context_recall)
        """
        required = {"question", "answer", "contexts"}
        missing = required - set(dataset.column_names)
        if missing:
            raise ValueError(f"Dataset 缺少必要列: {missing}")

        return await self._do_evaluate(dataset, metrics)

    # ------------------------------------------------------------------
    # 内部评估逻辑
    # ------------------------------------------------------------------
    async def _do_evaluate(
        self,
        dataset: Dataset,
        metrics: list[str] | None = None,
    ) -> EvalResult:
        """执行 RAGAS evaluate，返回结构化的 EvalResult"""
        from ragas import evaluate as ragas_evaluate

        # 1. 确定指标列表
        metric_names = metrics or ["faithfulness", "answer_relevancy", "context_precision"]
        if "ground_truth" in dataset.column_names and "context_recall" not in metric_names:
            metric_names.append("context_recall")

        # 2. 构建指标实例
        selected = _build_ragas_metrics(
            metric_names,
            llm=self._ragas_llm,
            embeddings=self._ragas_embeddings,
        )
        if not selected:
            raise ValueError(f"没有可用的评估指标: {metric_names}")

        logger.info(
            "RAGAS 评估开始  metrics=%s  samples=%d",
            [m.name for m in selected],
            len(dataset),
        )

        # 3. 执行评估（RAGAS evaluate 内部调用了 asyncio.run()，需放到独立线程避免嵌套事件循环冲突）
        try:
            result = await asyncio.to_thread(
                ragas_evaluate,
                dataset=dataset,
                metrics=selected,
            )
        except Exception as e:
            logger.error("RAGAS 评估失败: %s", e, exc_info=True)
            raise

        # 4. 解析 to_pandas()
        df = result.to_pandas()

        # 聚合分数
        score_cols = [c for c in df.columns if c not in ("question", "answer", "contexts", "ground_truth")]
        scores = {}
        for col in score_cols:
            try:
                scores[col] = round(float(df[col].mean()), 4)
            except (ValueError, TypeError):
                pass

        # 逐样本分数
        samples = []
        for _, row in df.iterrows():
            sample_scores = {}
            for col in score_cols:
                try:
                    sample_scores[col] = round(float(row[col]), 4)
                except (ValueError, TypeError):
                    pass
            samples.append(
                EvalSample(
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    scores=sample_scores,
                )
            )

        eval_result = EvalResult(
            metrics=list(scores.keys()),
            scores=scores,
            sample_count=len(dataset),
            samples=samples,
        )

        logger.info("RAGAS 评估完成: %s", scores)
        return eval_result
