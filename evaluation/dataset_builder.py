"""构建 RAGAS 评估使用的 HuggingFace Dataset

支持的数据源:
  1. 从业务数据库 (qa_records) 中提取
  2. 从 YAML/JSON 测试文件中读取
  3. 通过 RAG 系统实时问答构建（交互式评估）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.db_models import QARecord, get_session_maker
from utils.logger import logger


class EvalDatasetBuilder:
    """构建评估用数据集"""

    # ----------------------------------------------------------------
    # 从业务数据库构建
    # ----------------------------------------------------------------
    @staticmethod
    async def from_db(
        limit: int = 100,
        subject: str | None = None,
        user_id: str | None = None,
        min_feedback: int | None = None,
    ) -> Dataset:
        """从 qa_records 表构建评估数据集

        Args:
            limit: 最多提取的记录数
            subject: 按学科过滤
            user_id: 按用户过滤
            min_feedback: 1 仅好评, -1 仅差评, None 全部
        """
        session_maker = get_session_maker()
        async with session_maker() as session:
            query = select(QARecord).order_by(QARecord.created_at.desc()).limit(limit)
            if subject:
                query = query.where(QARecord.subject == subject)
            if user_id:
                query = query.where(QARecord.user_id == user_id)
            if min_feedback is not None:
                query = query.where(QARecord.feedback == min_feedback)

            result = await session.execute(query)
            records = result.scalars().all()

        questions, answers, contexts_list = [], [], []
        for r in records:
            if not r.answer or not r.retrieved_chunks:
                continue
            chunks = r.retrieved_chunks or []
            texts = [c.get("text", "") for c in chunks if c.get("text")]
            if not texts:
                continue
            questions.append(r.query)
            answers.append(r.answer)
            contexts_list.append(texts)

        logger.info(
            "从 QA 记录构建数据集: 共 %d 条记录, 有效样本 %d 条",
            len(records),
            len(questions),
        )
        return Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        })

    # ----------------------------------------------------------------
    # 从测试文件构建 (JSON / JSONL)
    # ----------------------------------------------------------------
    @staticmethod
    def from_file(file_path: str | Path) -> Dataset:
        """从 JSON 或 JSONL 文件加载测试集

        JSON 格式 (单对象 / 数组):
          {"question": "...", "answer": "...", "contexts": [...], "ground_truth": "..."}
          或 [{...}, ...]

        JSONL 格式 (每行一个对象):
          {"question": "...", "answer": "...", "contexts": [...], "ground_truth": "..."}
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"测试文件不存在: {file_path}")

        raw = path.read_text(encoding="utf-8").strip()
        if path.suffix == ".jsonl":
            items = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            items = json.loads(raw)
            if isinstance(items, dict):
                items = [items]

        return EvalDatasetBuilder._from_dicts(items)

    @staticmethod
    def from_dicts(items: list[dict]) -> Dataset:
        """从字典列表构建"""
        return EvalDatasetBuilder._from_dicts(items)

    @staticmethod
    def _from_dicts(items: list[dict]) -> Dataset:
        questions, answers, contexts_list, ground_truths = [], [], [], []
        for item in items:
            questions.append(item.get("question", ""))
            answers.append(item.get("answer", ""))
            ctx = item.get("contexts") or item.get("retrieved_docs") or item.get("context", [])
            contexts_list.append(ctx if isinstance(ctx, list) else [ctx])
            ground_truths.append(item.get("ground_truth", None))

        ds_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }
        if any(g is not None for g in ground_truths):
            ds_dict["ground_truth"] = [
                g if g else "" for g in ground_truths
            ]

        logger.info("从 dicts 构建数据集: %d 条样本", len(questions))
        return Dataset.from_dict(ds_dict)

    # ----------------------------------------------------------------
    # 手动构建
    # ----------------------------------------------------------------
    @staticmethod
    def from_manual(
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
        ground_truths: list[str | None] | None = None,
    ) -> Dataset:
        """直接传入 Q/A/Context 列表构建"""
        ds_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }
        if ground_truths:
            ds_dict["ground_truth"] = [g or "" for g in ground_truths]
        return Dataset.from_dict(ds_dict)
