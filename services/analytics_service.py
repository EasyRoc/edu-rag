"""学情分析服务：分析学生薄弱知识点、推荐复习内容"""

from sqlalchemy import select, func, desc

from models.db_models import QARecord, get_session_maker
from core.vectorestore import K12VectorStore
from utils.logger import logger


class AnalyticsService:
    """学情分析服务"""

    def __init__(self, vector_store: K12VectorStore):
        self.vector_store = vector_store

    async def get_weak_points(self, user_id: str, subject: str | None = None) -> list[dict]:
        """
        分析学生的薄弱知识点。

        策略：
        1. 获取该学生最近的问答记录（不限于差评，差评加权）
        2. 从每条记录的检索文档块中提取知识点（knowledge_point）
        3. 综合以下维度计算薄弱分数：
           - 差评占比：feedback == -1 的知识点权重高
           - 平均相关性：score 低说明相关知识不足
           - 出现频次：同一知识点反复出现说明是持续薄弱点
        4. 若 chunk 没有 knowledge_point，降级到 chapter；再没有则归为"通用"
        """
        logger.info(f"分析薄弱知识点: user_id={user_id}, subject={subject}")
        session_maker = get_session_maker()

        async with session_maker() as session:
            query = select(QARecord).where(QARecord.user_id == user_id)
            if subject:
                query = query.where(QARecord.subject == subject)
            query = query.order_by(desc(QARecord.created_at)).limit(50)
            result = await session.execute(query)
            records = result.scalars().all()

        # 按 (subject, chapter, knowledge_point) 聚合
        knowledge_map: dict[str, dict] = {}

        for record in records:
            chunks = record.retrieved_chunks or []
            for chunk in chunks:
                kp = chunk.get("knowledge_point", "") or ""
                ch = chunk.get("chapter", "") or ""
                subj = chunk.get("subject", record.subject or "未知")

                # 构建聚合键：有 knowledge_point 用它，否则用 chapter，再否则用 subject
                if kp:
                    key = f"{subj}||{ch}||{kp}"
                    label = kp
                elif ch:
                    key = f"{subj}||{ch}||"
                    label = ch
                else:
                    key = f"{subj}||||"
                    label = subj

                if key not in knowledge_map:
                    knowledge_map[key] = {
                        "subject": subj,
                        "chapter": ch or "通用",
                        "knowledge_point": kp or "通用",
                        "label": label,
                        "total_count": 0,
                        "neg_feedback_count": 0,
                        "total_score": 0.0,
                    }

                km = knowledge_map[key]
                km["total_count"] += 1
                km["total_score"] += chunk.get("score", 0)
                if record.feedback == -1:
                    km["neg_feedback_count"] += 1

        # 计算薄弱分数
        result_list = []
        for km in knowledge_map.values():
            avg_score = km["total_score"] / km["total_count"] if km["total_count"] > 0 else 0
            neg_ratio = km["neg_feedback_count"] / km["total_count"] if km["total_count"] > 0 else 0

            # 薄弱分数 = 差评比例 × 2.0 + (1 - 平均相关性) × 0.5
            # 差评占比越高越弱，相关性越低越弱
            km["weakness_score"] = round(neg_ratio * 2.0 + (1 - avg_score) * 0.5, 4)
            km["avg_score"] = round(avg_score, 4)
            result_list.append(km)

        result_list.sort(key=lambda x: x["weakness_score"], reverse=True)
        logger.info(f"薄弱知识点分析完成，发现 {len(result_list)} 个薄弱知识点")
        return result_list

    async def get_history(self, user_id: str, limit: int = 20) -> list[dict]:
        """获取学生问答历史"""
        logger.info(f"获取问答历史: user_id={user_id}, limit={limit}")
        session_maker = get_session_maker()
        async with session_maker() as session:
            query = (
                select(QARecord)
                .where(QARecord.user_id == user_id)
                .order_by(desc(QARecord.created_at))
                .limit(limit)
            )
            result = await session.execute(query)
            records = result.scalars().all()

        return [
            {
                "id": r.id,
                "query": r.query,
                "answer": r.answer[:200] if r.answer else "",
                "subject": r.subject,
                "grade": r.grade,
                "complexity": r.complexity,
                "feedback": r.feedback,
                "latency_ms": r.latency_ms,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in records
        ]

    async def recommend_review(self, user_id: str, subject: str | None = None) -> dict:
        """
        推荐复习内容。

        基于薄弱知识点，从知识库中检索相关材料。
        """
        weak_points = await self.get_weak_points(user_id, subject)
        if not weak_points:
            return {"message": "暂未发现薄弱知识点，继续保持！", "recommendations": []}
        # 取最薄弱学科的关键词进行检索
        weakest = weak_points[0]
        target_subject = weakest["subject"]
        related_docs = self.vector_store.hybrid_search(
            query=f"{target_subject} 基础知识 复习",
            subject=target_subject,
            grade=None,
            top_k=5,
        )
        recommendations = []
        for doc in related_docs:
            recommendations.append({
                "text": doc.get("text", "")[:300],
                "subject": doc.get("subject", target_subject),
                "grade": doc.get("grade", ""),
                "chapter": doc.get("chapter", ""),
                "knowledge_point": doc.get("knowledge_point", ""),
                "score": doc.get("score", 0),
            })
        return {
            "weak_subject": target_subject,
            "weak_knowledge_point": weakest.get("knowledge_point", "通用"),
            "weak_count": weakest["total_count"],
            "recommendations": recommendations,
        }
