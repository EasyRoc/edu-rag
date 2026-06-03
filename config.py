"""全局配置文件：从环境变量和 .env 文件中读取配置"""

import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 国内 HuggingFace 镜像配置
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")


class Settings:
    # ---------- LLM 配置（兼容 OpenAI API 格式）----------
    # 阿里百炼: https://dashscope.aliyuncs.com/compatible-mode/v1
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen-plus")
    # 模型上下文窗口上限（token 数），用于自动裁剪超长历史消息。
    # 默认 8192 是保守值，实际模型如 deepseek-v4-flash / qwen-plus 支持 128K+，
    # 可按模型实际窗口设置：LLM_MAX_CONTEXT_TOKENS=131072
    LLM_MAX_CONTEXT_TOKENS: int = int(os.getenv("LLM_MAX_CONTEXT_TOKENS", "8192"))

    # RAGAS / Instructor 结构化输出默认仅 1024 completion tokens，
    # faithfulness、context_* 等指标在长回答上易被截断，可通过环境变量提高上限。
    RAGAS_LLM_MAX_TOKENS: int = int(os.getenv("RAGAS_LLM_MAX_TOKENS", "8192"))

    # ---------- Milvus Lite 配置 ----------
    # 注意: 避免用 MILVUS_URI 命名（pymilvus 内部也读这个环境变量，会冲突）
    MILVUS_URI: str = os.getenv("K12_MILVUS_URI", "./milvus_k12.db")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "uploaded_docs"))

    # ---------- Embedding 配置 ----------
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # ---------- 意图识别配置 ----------
    LLM_TIMEOUT_SECONDS: int = 3         # LLM 分类调用超时
    ENABLE_LLM_FALLBACK: bool = True     # 是否启用 LLM 兜底

    # ---------- 应用配置 ----------
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ---------- 检索参数 ----------
    TOP_K: int = 5                # 检索返回 Top-K 结果
    CHUNK_SIZE: int = 512         # 文本切片大小
    CHUNK_OVERLAP: int = 64       # 切片重叠长度
    RRF_K: int = 60               # RRF 融合排名参数
    DENSE_WEIGHT: float = 0.7     # 稠密检索权重
    SPARSE_WEIGHT: float = 0.3    # 稀疏检索权重
    # 稠密检索（Milvus COSINE）：返回值为余弦相似度，越大越相似；低于此值的结果丢弃。0 表示不按阈值过滤。
    DENSE_MIN_SIMILARITY: float = float(os.getenv("DENSE_MIN_SIMILARITY", "0.0"))
    RETRIEVAL_CANDIDATE_TOP_K: int = int(os.getenv("RETRIEVAL_CANDIDATE_TOP_K", "20"))
    GENERATION_CONTEXT_TOP_K: int = int(os.getenv("GENERATION_CONTEXT_TOP_K", "5"))

    # ---------- 本地重排与检索门控 ----------
    ENABLE_RERANKER: bool = os.getenv("ENABLE_RERANKER", "true").lower() in {"1", "true", "yes", "on"}
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    RERANKER_DEVICE: str = os.getenv("RERANKER_DEVICE", "cpu")
    RERANKER_BATCH_SIZE: int = int(os.getenv("RERANKER_BATCH_SIZE", "16"))
    RERANKER_RELEVANCE_THRESHOLD: float = float(os.getenv("RERANKER_RELEVANCE_THRESHOLD", "0.50"))
    RETRIEVAL_ACCEPT_TOP1_THRESHOLD: float = float(os.getenv("RETRIEVAL_ACCEPT_TOP1_THRESHOLD", "0.60"))
    RETRIEVAL_GATE_MODE: str = os.getenv("RETRIEVAL_GATE_MODE", "enforce")

    # ---------- 多策略检索 ----------
    MULTI_QUERY_VARIANTS: int = int(os.getenv("MULTI_QUERY_VARIANTS", "4"))       # 多查询生成的变体数量
    DECOMPOSITION_MAX_SUB: int = int(os.getenv("DECOMPOSITION_MAX_SUB", "4"))     # 复杂问题最多拆解的子问题数
    STRATEGY_TIMEOUT: float = float(os.getenv("STRATEGY_TIMEOUT", "10"))           # 策略 LLM 调用超时(秒)

    # ---------- 纠正重试 ----------
    MAX_RETRIES: int = max(0, min(2, int(os.getenv("MAX_RETRIES", "2"))))  # Corrective RAG 最大重试次数

    # ---------- Milvus 集合名称 ----------
    MILVUS_COLLECTION: str = "k12_knowledge_base"

    # ---------- SQLite 数据库路径 ----------
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{BASE_DIR}/k12_business.db")


settings = Settings()
