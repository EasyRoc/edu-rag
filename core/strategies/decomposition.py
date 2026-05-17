"""复杂问题分解策略：将复杂查询拆解为子问题，分别检索后合并"""
from config import settings
from core.strategies._llm import llm_complete
from utils.logger import logger

DECOMPOSE_SYSTEM = """你是一个问题分析助手。请将复杂问题拆解为 2-4 个简单的子问题，每个子问题应能独立检索和回答。

规则：
1. 每个子问题聚焦单一知识点
2. 子问题之间尽量不重叠
3. 子问题合起来能覆盖原问题的全部要点
4. 每行一个子问题，不要编号，不要额外解释"""


async def decompose_query(query: str, max_sub: int | None = None) -> list[str]:
    """将复杂问题拆解为子问题列表，失败时返回原始问题"""
    if max_sub is None:
        max_sub = getattr(settings, 'DECOMPOSITION_MAX_SUB', 4)

    user_prompt = f"复杂问题：{query}\n\n请拆解为子问题："
    result = await llm_complete(DECOMPOSE_SYSTEM, user_prompt)

    if not result:
        logger.warning("问题分解失败，降级为原始问题")
        return [query]

    sub_queries = []
    for line in result.strip().split("\n"):
        line = line.strip()
        for prefix in ["- ", "· "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
        import re
        line = re.sub(r'^\d+[\.\、\)]\s*', '', line)
        if line and len(line) > 2:
            sub_queries.append(line)

    if not sub_queries:
        return [query]

    logger.info(f"问题分解: 拆解为 {len(sub_queries)} 个子问题")
    return sub_queries[:max_sub]


def merge_sub_results(sub_results: list[list[dict]], top_k: int) -> list[dict]:
    """合并子问题检索结果，按 score 去重排序"""
    if not sub_results:
        return []

    seen: dict[int, dict] = {}
    for result_list in sub_results:
        for doc in result_list:
            chunk_key = doc.get("id", 0)
            if chunk_key not in seen or doc["score"] > seen[chunk_key]["score"]:
                seen[chunk_key] = doc

    merged = sorted(seen.values(), key=lambda d: d["score"], reverse=True)
    logger.info(f"子结果合并: {len(sub_results)} 组 → {len(merged)} 条（去重后）")
    return merged[:top_k]
