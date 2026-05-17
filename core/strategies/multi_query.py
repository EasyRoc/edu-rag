"""多查询策略：生成多个查询变体，各路检索后用 RRF 融合"""
from collections import defaultdict
from config import settings
from core.strategies._llm import llm_complete
from utils.logger import logger

MULTI_QUERY_SYSTEM = """你是一个查询改写助手。请将用户的问题改写为多个不同角度的同义表述，用于提升文档检索的召回率。

规则：
1. 保持原问题的核心意图不变
2. 从不同角度、用不同措辞表达同一问题
3. 每行一个改写结果，不要编号，不要额外解释"""


async def generate_query_variants(query: str, n: int | None = None) -> list[str]:
    """生成 n 个查询变体，失败时返回空列表"""
    if n is None:
        n = getattr(settings, 'MULTI_QUERY_VARIANTS', 4)

    user_prompt = f"原始问题：{query}\n\n请生成 {n} 个改写："
    result = await llm_complete(MULTI_QUERY_SYSTEM, user_prompt)

    if not result:
        logger.warning("多查询变体生成失败，降级为空列表")
        return []

    # 按行解析，过滤空行和原始问题
    variants = []
    for line in result.strip().split("\n"):
        line = line.strip()
        # 去除可能的编号前缀: "1. xxx" / "1、xxx" / "- xxx"
        for prefix in ["- ", "· "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
        # 去除数字编号
        import re
        line = re.sub(r'^\d+[\.\、\)]\s*', '', line)
        if line and line != query and len(line) > 2:
            variants.append(line)

    logger.info(f"多查询变体: 生成了 {len(variants)} 个变体")
    return variants[:n]


def multi_query_fusion(all_results: list[list[dict]], top_k: int, rrf_k: int = 60) -> list[dict]:
    """多查询结果 RRF 融合：对各路检索结果统一排序去重"""
    if not all_results:
        return []

    score_map: dict[int, float] = defaultdict(float)
    doc_map: dict[int, dict] = {}

    for result_list in all_results:
        for rank, doc in enumerate(result_list):
            chunk_key = doc.get("id", 0)
            score_map[chunk_key] += 1.0 / (rrf_k + rank + 1)
            if chunk_key not in doc_map:
                doc_map[chunk_key] = doc

    # 按 RRF 得分排序
    scored = [(score_map[chunk_key], doc_map[chunk_key]) for chunk_key in doc_map]
    scored.sort(key=lambda x: x[0], reverse=True)

    # 归一化（复制避免修改原始 dict）
    max_score = scored[0][0] if scored else 1
    results = []
    for score, doc in scored[:top_k]:
        doc = dict(doc)
        doc["score"] = round(score / max_score, 4)
        results.append(doc)

    logger.info(f"多查询融合: {len(all_results)} 路结果 → {len(results)} 条")
    return results
