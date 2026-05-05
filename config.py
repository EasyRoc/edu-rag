"""全局配置文件：从环境变量和 .env 文件中读取配置"""

import os
from dotenv import load_dotenv

load_dotenv()

# 国内 HuggingFace 镜像配置
if os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")


class Settings:
    # ---------- LLM 配置（兼容 OpenAI API 格式）----------
    # 阿里百炼: https://dashscope.aliyuncs.com/compatible-mode/v1
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen-plus")

    # ---------- Milvus Lite 配置 ----------
    # 注意: 避免用 MILVUS_URI 命名（pymilvus 内部也读这个环境变量，会冲突）
    MILVUS_URI: str = os.getenv("K12_MILVUS_URI", "./milvus_k12.db")

    # ---------- Embedding 配置 ----------
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

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

    # ---------- 纠正重试 ----------
    MAX_RETRIES: int = 2           # Corrective RAG 最大重试次数

    # ---------- Milvus 集合名称 ----------
    MILVUS_COLLECTION: str = "k12_knowledge_base"

    # ---------- SQLite 数据库路径 ----------
    DATABASE_URL: str = f"sqlite+aiosqlite:///{os.path.dirname(os.path.abspath(__file__))}/k12_business.db"


settings = Settings()
