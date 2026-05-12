"""测试集生成器：LLM 辅助生成 RAGAS 评估用测试集

支持:
  1. 从向量库文档片段生成 question + ground_truth
  2. 从 QA 历史记录中筛选并补全 ground_truth
  3. 测试集校验（去重、分布统计）
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from config import settings
from core.vectorestore import K12VectorStore
from models.db_models import QARecord, get_session_maker
from sqlalchemy import select
from utils.logger import logger

# ---------------------------------------------------------------------------
# LLM Prompt 模板
# ---------------------------------------------------------------------------

GEN_QUESTIONS_PROMPT = """你是一个 K12 教育测试专家。根据下面提供的知识点内容，生成高质量的问题和参考答案，用于评估 RAG 问答系统的质量。

## 知识点内容
{context}

## 学科: {subject}  年级段: {grade}

## 生成要求
请生成 {count} 道不同类型的问题，覆盖以下三个难度级别：

1. **simple（事实检索）**：直接询问知识点中的事实、定义、公式。答案可在原文中直接找到。
2. **medium（概念解释）**：需要对知识点进行理解、解释、举例说明。
3. **complex（综合推理）**：需要综合多个知识点、对比分析或解决实际问题。

每道题以 JSON 格式输出，包含：
- question: 问题文本
- ground_truth: 标准参考答案
- complexity: simple / medium / complex
- question_type: 问题类型（定义题/计算题/应用题/对比题/开放题）

## 输出格式
只输出 JSON 数组，不要任何其他内容：
```json
[
  {{"question": "...", "ground_truth": "...", "complexity": "simple", "question_type": "定义题"}},
  ...
]
```"""

GEN_GROUND_TRUTH_PROMPT = """根据以下问答记录，为该问题撰写一份标准参考答案（ground_truth）。
只输出答案文本，不要任何前缀或说明。

## 问题
{question}

## 系统生成的回答（仅供参考）
{answer}

## 检索到的上下文
{contexts}

## 标准答案
"""


# ---------------------------------------------------------------------------
# TestSetGenerator
# ---------------------------------------------------------------------------

class TestSetGenerator:
    """LLM 辅助测试集生成器"""

    def __init__(self):
        self._api_key = settings.LLM_API_KEY
        self._base_url = settings.LLM_BASE_URL.rstrip("/")
        self._model = settings.LLM_MODEL

    # ------------------------------------------------------------------
    # 方式 B：从向量库文档生成
    # ------------------------------------------------------------------
    async def from_vectorestore(
        self,
        vector_store: K12VectorStore,
        subject: str | None = None,
        grade: str | None = None,
        count: int = 30,
    ) -> list[dict]:
        """从向量库中随机采样文档片段，用 LLM 生成问题 + ground_truth

        每段文档生成 3 题（每个复杂度 1 题），多段合并达到 count 数量。
        """
        # 1. 从向量库采样文档
        docs = self._sample_docs(vector_store, subject, grade, count=count // 3 + 1)
        if not docs:
            logger.warning("向量库中没有匹配的文档")
            return []

        items = []
        for doc in docs:
            doc_text = doc.get("text", "")
            doc_subject = doc.get("subject", subject or "")
            doc_grade = doc.get("grade", grade or "")
            if not doc_text or len(doc_text) < 50:
                continue

            try:
                generated = await self._generate_from_doc(
                    context=doc_text,
                    subject=doc_subject,
                    grade=doc_grade,
                    count=3,
                )
                items.extend(generated)
            except Exception as e:
                logger.warning(f"单个文档生成失败: {e}")

        return items[:count]

    # ------------------------------------------------------------------
    # 方式 C：从 QA 历史筛选并补全 ground_truth
    # ------------------------------------------------------------------
    async def from_qa_history(
        self,
        limit: int = 50,
        subject: str | None = None,
        feedback: int | None = 1,
    ) -> list[dict]:
        """从 QA 记录中筛选样本，用 LLM 补全 ground_truth"""
        session_maker = get_session_maker()
        async with session_maker() as session:
            query = select(QARecord).order_by(QARecord.created_at.desc()).limit(limit)
            if subject:
                query = query.where(QARecord.subject == subject)
            if feedback is not None:
                query = query.where(QARecord.feedback == feedback)
            result = await session.execute(query)
            records = result.scalars().all()

        items = []
        for r in records:
            if not r.answer or not r.query:
                continue
            chunks = r.retrieved_chunks or []
            ctx_texts = [c.get("text", "") for c in chunks if c.get("text")]
            ctx_str = "\n---\n".join(ctx_texts[:5])

            ground_truth = await self._generate_ground_truth(
                question=r.query,
                answer=r.answer,
                contexts=ctx_str,
            )
            items.append({
                "question": r.query,
                "ground_truth": ground_truth or "",
                "subject": r.subject or "",
                "grade": r.grade or "",
            })

        return items

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    @staticmethod
    def validate(items: list[dict]) -> dict:
        """校验测试集：统计分布、检测空字段、去重"""
        report: dict[str, Any] = {
            "total": len(items),
            "duplicates_removed": 0,
            "missing_question": 0,
            "missing_ground_truth": 0,
            "empty_contexts": 0,
            "complexity_dist": Counter(),
            "subject_dist": Counter(),
            "grade_dist": Counter(),
            "question_type_dist": Counter(),
        }

        seen = set()
        deduped = []
        for item in items:
            q = (item.get("question") or "").strip()
            if not q:
                report["missing_question"] += 1
                continue
            key = q.lower()
            if key in seen:
                report["duplicates_removed"] += 1
                continue
            seen.add(key)

            if not item.get("ground_truth"):
                report["missing_ground_truth"] += 1
            if not item.get("contexts"):
                report["empty_contexts"] += 1
            report["complexity_dist"][item.get("complexity", "unknown")] += 1
            report["subject_dist"][item.get("subject", "unknown")] += 1
            report["grade_dist"][item.get("grade", "unknown")] += 1
            report["question_type_dist"][item.get("question_type", "unknown")] += 1
            deduped.append(item)

        report["total_after_dedup"] = len(deduped)
        report["complexity_dist"] = dict(report["complexity_dist"])
        report["subject_dist"] = dict(report["subject_dist"])
        report["grade_dist"] = dict(report["grade_dist"])
        report["question_type_dist"] = dict(report["question_type_dist"])
        return report

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    @staticmethod
    def save(items: list[dict], path: str | Path) -> str:
        """保存测试集为 JSONL 文件，每行一个 JSON 对象"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"测试集已保存: {path} ({len(items)} 条)")
        return str(path)

    # ------------------------------------------------------------------
    # 内部：向量库采样
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_docs(
        vector_store: K12VectorStore,
        subject: str | None,
        grade: str | None,
        count: int = 30,
    ) -> list[dict]:
        """从向量库随机采样文档片段"""
        col = vector_store.collection
        expr_parts = ['id != ""']
        if subject:
            expr_parts.append(f'subject == "{subject}"')
        if grade:
            expr_parts.append(f'grade == "{grade}"')
        expr = " and ".join(expr_parts)

        results = col.query(
            expr=expr,
            output_fields=["text", "subject", "grade", "doc_id"],
            limit=min(count, 100),
        )
        return results

    # ------------------------------------------------------------------
    # 内部：LLM 生成问题
    # ------------------------------------------------------------------
    async def _generate_from_doc(
        self,
        context: str,
        subject: str,
        grade: str,
        count: int = 3,
    ) -> list[dict]:
        """用 LLM 从文档片段生成问题"""
        prompt = GEN_QUESTIONS_PROMPT.format(
            context=context[:3000],
            subject=subject,
            grade=grade,
            count=count,
        )
        raw = await self._call_llm(prompt)
        return self._parse_json_response(raw)

    # ------------------------------------------------------------------
    # 内部：LLM 生成 ground_truth
    # ------------------------------------------------------------------
    async def _generate_ground_truth(
        self,
        question: str,
        answer: str,
        contexts: str,
    ) -> str | None:
        """用 LLM 补全 ground_truth"""
        prompt = GEN_GROUND_TRUTH_PROMPT.format(
            question=question,
            answer=answer,
            contexts=contexts[:3000],
        )
        try:
            gt = await self._call_llm(prompt)
            return gt.strip()
        except Exception as e:
            logger.warning(f"生成 ground_truth 失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 内部：LLM 调用
    # ------------------------------------------------------------------
    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM（非流式）"""
        if not self._api_key:
            raise RuntimeError("未配置 LLM_API_KEY")

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": "你是一个专业的 K12 教育评估专家。"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
            )
            resp.raise_for_status()
            result = resp.json()
            return result["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # 内部：解析 LLM JSON 响应
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json_response(raw: str) -> list[dict]:
        """从 LLM 响应中提取 JSON 数组"""
        # 尝试直接解析
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 代码块
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取 [...] 数组
        m = re.search(r'\[[\s\S]*\]', raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"无法解析 LLM 响应为 JSON: {raw[:200]}")
        return []
