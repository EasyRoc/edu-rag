"""评估流水线：支持 API 调用和 CLI 两种模式

流程:
  1. 构建测试数据集 (from DB / from file / from manual)
  2. 对每个样本执行 RAG 流程获取 answer + contexts
  3. 用 RAGAS 评估
  4. 保存结果到数据库
  5. 输出评估报告
"""

from __future__ import annotations

import time
from datetime import datetime

from datasets import Dataset

from config import settings
from core.graph import build_rag_graph
from core.vectorestore import K12VectorStore
from evaluation.ragas_evaluator import RAGASEvaluator
from evaluation.schemas import EvalResult, eval_result_to_dict
from models.db_models import EvaluationRecord, get_session_maker
from services.rag_service import RAGService
from utils.logger import logger


# ======================================================================
# 执行一次完整的离线评估
# ======================================================================

async def run_evaluation(
    dataset: Dataset,
    name: str = "default",
    metrics: list[str] | None = None,
    save_to_db: bool = True,
) -> EvalResult:
    """执行完整评估流水线

    Args:
        dataset: 评估数据集 (含 question, contexts, 可选 ground_truth)
        name: 评估任务名称，用于标识
        metrics: 评估指标列表，默认 ["faithfulness", "answer_relevancy", "context_precision"]
        save_to_db: 是否将结果持久化到业务数据库

    Returns:
        EvalResult 包含聚合评分和每个样本的得分
    """
    logger.info("========== 开始 RAGAS 评估 ==========")
    logger.info("评估任务: %s", name)
    logger.info("样本数: %d", len(dataset))
    logger.info("指标: %s", metrics or ["faithfulness", "answer_relevancy", "context_precision"])

    # 1. 初始化 Evaluator
    evaluator = RAGASEvaluator()

    # 2. 执行评估
    start = time.time()
    try:
        result = await evaluator.evaluate_dataset(dataset, metrics=metrics)
    except Exception as e:
        logger.error("RAGAS 评估失败: %s", e, exc_info=True)
        raise

    elapsed = time.time() - start

    # 3. 附加元信息
    result.extra = {
        "name": name,
        "timestamp": datetime.utcnow().isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "sample_count": len(dataset),
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
    }

    # 4. 持久化
    if save_to_db:
        await _save_eval_result(result)

    # 5. 打印报告
    _print_report(result)

    logger.info("========== RAGAS 评估完成 (%.2fs) ==========", elapsed)
    return result


# ======================================================================
# 实时评估：先用 RAG 系统回答问题，再评估
# ======================================================================

async def run_live_evaluation(
    questions: list[str],
    vector_store: K12VectorStore,
    subject: str | None = None,
    grade: str | None = None,
    metrics: list[str] | None = None,
    name: str = "live_eval",
    ground_truths: list[str | None] | None = None,
) -> EvalResult:
    """实时问答 + RAGAS 评估

    对给定的问题列表，调用 RAG 系统依次回答，然后评估回答质量。
    适用于上线前测试或夜间回归。
    """
    logger.info("实时评估模式: %d 个问题", len(questions))

    # 构建 RAG 图
    rag_graph = build_rag_graph(vector_store)
    service = RAGService(vector_store, rag_graph)

    # 依次问答
    answers = []
    contexts_list = []
    for i, q in enumerate(questions):
        logger.info("[%d/%d] 问答: %s", i + 1, len(questions), q[:80])
        resp = await service.ask(
            query=q,
            subject=subject,
            grade=grade,
            user_id="__evaluator__",
        )
        answers.append(resp.get("answer", ""))
        refs = resp.get("references", [])
        contexts_list.append([r.get("text", "") for r in refs if r.get("text")])

    # 构建 Dataset
    ds_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
    }
    if ground_truths:
        ds_dict["ground_truth"] = [g or "" for g in ground_truths]
    dataset = Dataset.from_dict(ds_dict)

    # 评估
    return await run_evaluation(dataset, name=name, metrics=metrics, vector_store=vector_store)


# ======================================================================
# 结果持久化
# ======================================================================

async def _save_eval_result(result: EvalResult) -> str | None:
    """将评估结果存入 EvaluationRecord 表"""
    try:
        session_maker = get_session_maker()
        async with session_maker() as session:
            record = EvaluationRecord(
                task_name=result.extra.get("name", "unnamed"),
                metrics=result.metrics,
                scores=result.scores,
                sample_count=result.sample_count,
                samples=[
                    {
                        "question": s.question[:500],
                        "answer": s.answer[:500],
                        "scores": s.scores,
                    }
                    for s in result.samples
                ],
                config_snapshot={
                    "llm_model": settings.LLM_MODEL,
                    "llm_base_url": settings.LLM_BASE_URL,
                    "embedding_model": settings.EMBEDDING_MODEL,
                    "top_k": settings.TOP_K,
                    "rrf_k": settings.RRF_K,
                    "dense_weight": settings.DENSE_WEIGHT,
                    "sparse_weight": settings.SPARSE_WEIGHT,
                },
                elapsed_seconds=result.extra.get("elapsed_seconds", 0),
            )
            session.add(record)
            await session.commit()
            logger.info("评估结果已保存到数据库: id=%s", record.id)
            return record.id
    except Exception as e:
        logger.warning("保存评估结果到数据库失败: %s", e)
        return None


# ======================================================================
# 报告打印
# ======================================================================

def _print_report(result: EvalResult) -> None:
    """打印格式化评估报告"""
    sep = "=" * 50
    print(f"\n{sep}")
    print("  RAGAS 评估报告")
    print(f"  任务: {result.extra.get('name', 'N/A')}")
    print(f"  时间: {result.extra.get('timestamp', 'N/A')}")
    print(f"  样本数: {result.sample_count}")
    print(f"{sep}")
    print(f"  聚合得分:")
    for metric, score in sorted(result.scores.items()):
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {metric:<30s} {score:.4f}  {bar}")
    print(f"{sep}")
    print(f"  LLM 模型: {result.extra.get('llm_model', 'N/A')}")
    print(f"  Embedding: {result.extra.get('embedding_model', 'N/A')}")
    print(f"  耗时: {result.extra.get('elapsed_seconds', 0):.1f}s")
    print(f"{sep}\n")
