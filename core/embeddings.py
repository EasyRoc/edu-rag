from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config import settings
from utils.logger import logger

# 全局单例（避免重复加载模型）
_embedding_model = None


# 获取向量模型
def get_embedding_model() -> HuggingFaceBgeEmbeddings:
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    logger.info(f"正在加载 Embedding 模型: {settings.EMBEDDING_MODEL}")
    logger.info(f"使用设备: {settings.EMBEDDING_DEVICE}")

    model_kwargs = {"device": settings.EMBEDDING_DEVICE}
    encode_kwargs = {"normalize_embeddings": True}
    try:
        _embedding_model = HuggingFaceBgeEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_instruction="为这个句子生成表示以用于检索相关文章：",
        )
        logger.info(f"Embedding 模型加载成功")
    except Exception as e:
        logger.error(f"Embedding 模型加载失败: {e}")
        raise

    return _embedding_model


def get_embedding_dim() -> int:
    """获取 Embedding 模型的输出向量维度"""
    model = get_embedding_model()
    # 通过编码一个测试文本获取维度
    test_vec = model.embed_query("测试")
    return len(test_vec)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """将文本列表批量转换为向量"""
    model = get_embedding_model()
    logger.debug(f"正在对 {len(texts)} 条文本进行向量化")
    vectors = model.embed_documents(texts)
    logger.debug(f"向量化完成，向量维度: {len(vectors[0]) if vectors else 0}")
    return vectors


def embed_query(text: str) -> list[float]:
    model = get_embedding_model()
    logger.debug(f"正在对查询进行向量化: {text[:50]}...")
    vector = model.embed_query(text)
    return vector
