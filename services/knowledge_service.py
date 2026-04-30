"""知识点管理服务：知识点层级树的 CRUD"""

from models.db_models import KnowledgePoint, get_session_maker
from sqlalchemy import select
from utils.logger import logger


class KnowledgeService:
    """知识点管理服务"""

    async def create_knowledge_point(self, data: dict) -> dict:
        """创建知识点"""
        session_maker = get_session_maker()
        async with session_maker() as session:
            # 计算层级
            level = 0
            if data.get("parent_id"):
                parent = await session.get(KnowledgePoint, data["parent_id"])
                if parent:
                    level = parent.level + 1

            kp = KnowledgePoint(
                name=data["name"],
                subject=data["subject"],
                parent_id=data.get("parent_id"),
                level=level,
                description=data.get("description", ""),
                sort_order=data.get("sort_order", 0),
            )
            session.add(kp)
            await session.commit()
            await session.refresh(kp)
            logger.info(f"知识点已创建: {kp.name} (subject={kp.subject}, level={kp.level})")
            return {
                "id": kp.id,
                "name": kp.name,
                "subject": kp.subject,
                "parent_id": kp.parent_id,
                "level": kp.level,
                "description": kp.description,
            }

    async def get_knowledge_tree(self, subject: str | None = None) -> list[dict]:
        """获取知识点树（支持按学科过滤）"""
        session_maker = get_session_maker()
        async with session_maker() as session:
            query = select(KnowledgePoint).order_by(KnowledgePoint.sort_order)
            if subject:
                query = query.where(KnowledgePoint.subject == subject)
            result = await session.execute(query)
            all_kps = result.scalars().all()

        # 构建树形结构
        kp_map = {}
        for kp in all_kps:
            kp_map[kp.id] = {
                "id": kp.id,
                "name": kp.name,
                "subject": kp.subject,
                "parent_id": kp.parent_id,
                "level": kp.level,
                "description": kp.description,
                "children": [],
            }

        tree = []
        for kp_id, node in kp_map.items():
            if node["parent_id"] and node["parent_id"] in kp_map:
                kp_map[node["parent_id"]]["children"].append(node)
            else:
                tree.append(node)

        return tree

    async def delete_knowledge_point(self, kp_id: str) -> bool:
        """删除知识点"""
        session_maker = get_session_maker()
        async with session_maker() as session:
            kp = await session.get(KnowledgePoint, kp_id)
            if not kp:
                return False
            await session.delete(kp)
            await session.commit()
            logger.info(f"知识点已删除: {kp_id}")
            return True
