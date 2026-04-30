"""检索节点：根据查询复杂度执行不同深度和广度的混合检索"""
from core.vectorestore import K12VectorStore
from utils.logger import logger


def hybrid_retrieve(
    vector_store: K12VectorStore,
    query: str,
    complexity: str,
    subject: str | None = None,
    grade: str | None = None,
) -> list[dict]:
    """
    混合检索节点。

    根据查询复杂度调整检索策略：
    - simple: 只检索 top_k=3，提升速度
    - medium: 标准检索 top_k=5
    - complex: 扩大检索 top_k=8，获取更多上下文
    """
    top_k_map = {"simple": 3, "medium": 5, "complex": 8}
    top_k = top_k_map.get(complexity, 5)
    logger.info(f"检索节点: complexity={complexity}, top_k={top_k}")
    docs = vector_store.hybrid_search(
        query=query,
        subject=subject,
        grade=grade,
        top_k=top_k,
    )
    logger.info(f"检索完成，获取到 {len(docs)} 篇相关文档")
    for i, doc in enumerate(docs):
        logger.debug(f"  结果[{i}]: score={doc['score']:.4f}, source={doc.get('_source', '?')}, "
                     f"text={doc['text'][:60]}...")

    return docs