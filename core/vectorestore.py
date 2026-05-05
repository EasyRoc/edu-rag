import numpy as np
from pymilvus import MilvusClient, DataType
from rank_bm25 import BM25Okapi
from core.embeddings import get_embedding_model, get_embedding_dim, embed_texts, embed_query
from utils.logger import logger
from config import settings


class K12VectorStore:
    """
    K12 混合向量存储：
    - 稠密检索：Milvus Lite（ANN 搜索）
    - 稀疏检索：本地 BM25（关键词搜索）
    - 融合策略：RRF（倒数排名融合）
    """

    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.embedding_dim = get_embedding_dim()  # 动态获取模型输出维度

        # ---------- Milvus Lite 客户端 ----------
        logger.info(f"初始化 Milvus Lite，数据文件: {settings.MILVUS_URI}")
        self.milvus_client = MilvusClient(settings.MILVUS_URI)
        self._init_collection()

        # ---------- 本地 BM25 组件 ----------
        self.bm25: BM25Okapi | None = None
        self.bm25_docs: list[dict] = []  # 与 BM25 对应的文档列表
        self.bm25_corpus: list[list[str]] = []  # 分词后的语料库

        logger.info("K12VectorStore 初始化完成")

    # ==================== Milvus 集合管理 ====================

    def _init_collection(self):
        collection_name = settings.MILVUS_COLLECTION
        if self.milvus_client.has_collection(collection_name):
            logger.info(f"集合 '{collection_name}' 已存在，直接加载")
            self.milvus_client.load_collection(collection_name)
            self._ensure_index()
            return
        logger.info(f"正在创建集合: {collection_name}")
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_text", DataType.VARCHAR, max_length=8192)
        schema.add_field("subject", DataType.VARCHAR, max_length=32)
        schema.add_field("grade", DataType.VARCHAR, max_length=32)
        schema.add_field("chapter", DataType.VARCHAR, max_length=128)
        schema.add_field("knowledge_point", DataType.VARCHAR, max_length=128)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=32)
        self.milvus_client.create_collection(collection_name, schema=schema)
        # 创建索引（与 create_collection 分开调用，确保生效）
        self._ensure_index()
        logger.info(f"集合 '{collection_name}' 创建成功")

    def _ensure_index(self):
        """确保向量字段上有索引（Milvus Lite 需要先创建集合再建索引）"""
        collection_name = settings.MILVUS_COLLECTION
        logger.info(f"确保索引存在: {collection_name}")

        try:
            existing = self.milvus_client.list_indexes(collection_name)
            if existing:
                logger.info(f"索引已存在: {existing}")
                return
        except ImportError:
            pass
        index_params = self.milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        self.milvus_client.create_index(collection_name, index_params)
        logger.info("向量索引创建成功")

    def insert_chunks(self, chunks: list[dict]) -> list[int]:
        """
        将切片列表插入向量存储。

        chunks: 每个元素包含 text, doc_id, subject, grade, chapter, knowledge_point, chunk_type
        返回插入的 Milvus ID 列表。
        """
        if not chunks:
            logger.warning("插入的切片列表为空")
            return []
        collection_name = settings.MILVUS_COLLECTION
        logger.info(f"正在插入 {len(chunks)} 个切片到 Milvus")
        # 1. 批量生成向量
        texts = [c["text"] for c in chunks]
        vectors = embed_texts(texts)

        # 2. 准备 Milvus 数据
        milvus_data = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            milvus_data.append({
                "vector": vec,
                "doc_id": chunk.get("doc_id", ""),
                "chunk_text": chunk["text"],
                "subject": chunk.get("subject", ""),
                "grade": chunk.get("grade", ""),
                "chapter": chunk.get("chapter", ""),
                "knowledge_point": chunk.get("knowledge_point", ""),
                "chunk_type": chunk.get("chunk_type", "text"),
            })
        ids = self.milvus_client.insert(collection_name=collection_name,
                                        data=milvus_data, )
        logger.info(f"Milvus 插入完成，共 {len(ids)} 条")
        # 4. 同步更新本地 BM25 索引
        self._rebuild_bm25_index()

        return ids

    def _rebuild_bm25_index(self):
        """
        从 Milvus 中读取所有数据，重建本地 BM25 索引。
        在插入或删除后调用。
        """
        logger.info("正在重建 BM25 索引...")
        collection_name = settings.MILVUS_COLLECTION
        try:
            result = self.milvus_client.query(
                collection_name=collection_name,
                filter="",
                output_fields=["id", "chunk_text", "subject", "grade", "chapter", "knowledge_point"],
                limit=10000,
            )
        except Exception as e:
            logger.warning(f"查询 Milvus 数据失败，BM25 索引可能为空: {e}")
            self.bm25_docs = []
            self.bm25_corpus = []
            self.bm25 = None
            return
        if not result:
            self.bm25_docs = []
            self.bm25_corpus = []
            self.bm25 = None
            logger.info("BM25 索引重建完成，共 0 篇文档（集合为空）")
            return
        self.bm25_docs = result
        self.bm25_corpus = [self._tokenize(doc["chunk_text"]) for doc in result]
        self.bm25 = BM25Okapi(self.bm25_corpus)
        logger.info(f"BM25 索引重建完成，共 {len(result)} 篇文档")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单中文分词（按字/词切分，实际项目可使用 jieba）"""
        # 简单实现：按空格和标点切分后，再按字符二元组切分
        import re
        tokens = re.findall(r"[\w]+", text)
        result = []
        for token in tokens:
            if len(token) <= 2:
                result.append(token)
            else:
                # 二元组切分，增强中文检索效果
                for i in range(len(token) - 1):
                    result.append(token[i:i + 2])
        return result

    # ==================== 混合检索 ====================

    def hybrid_search(
            self,
            query: str,
            subject: str | None = None,
            grade: str | None = None,
            top_k: int = 5,
    ) -> list[dict]:
        """
        混合检索：稠密向量 + 稀疏 BM25，RRF 融合。

        返回按相关性降序排列的文档列表，每项包含 score 和元数据。
        """
        logger.info(f"混合检索: query='{query[:50]}', subject={subject}, grade={grade}, top_k={top_k}")
        filters = []
        if subject:
            filters.append(f"subject == '{subject}'")
        if grade:
            filters.append(f"grade == '{grade}'")
        filter_str = " and ".join(filters) if filters else ""
        logger.info(f"混合检索 filter_str=={filter_str}")

        # 1. 稠密检索（Milvus ANN）
        dense_results = self._dense_search(query, filter_str, top_k)
        logger.debug(f"稠密检索返回 {len(dense_results)} 条结果")

        # 2. 稀疏检索（本地 BM25）
        sparse_results = self._sparse_search(query, filter_str, top_k)
        logger.debug(f"稀疏检索返回 {len(sparse_results)} 条结果")

        # 3. RRF 融合
        if not dense_results and not sparse_results:
            logger.warning("稠密和稀疏检索均无结果")
            return []
        if not dense_results:
            return sparse_results[:top_k]
        if not sparse_results:
            return dense_results[:top_k]

        fused = self._rrf_fusion(dense_results, sparse_results, top_k)
        logger.info(f"混合检索完成，最终返回 {len(fused)} 条结果")
        return fused

    def _dense_search(self, query: str, filter_str: str, top_k: int) -> list[dict]:
        """Milvus 稠密向量 ANN 搜索"""
        query_vec = embed_query(query)
        collection_name = settings.MILVUS_COLLECTION
        results = self.milvus_client.search(
            collection_name=collection_name,
            data=[query_vec],
            filter=filter_str or None,
            limit=top_k,
            output_fields=["chunk_text", "doc_id", "subject", "grade", "chapter", "knowledge_point"],
            search_params={"metric_type": "COSINE", "params": {"ef": 10}},
        )
        if not results or not results[0]:
            return []
        docs = []
        for hit in results[0]:
            # Milvus metric_type=COSINE 时，distance 为余弦相似度，越大越相关（通常约 0~1）
            sim = float(hit["distance"])
            docs.append({
                "id": hit["id"],
                "text": hit["entity"]["chunk_text"],
                "score": sim,
                "doc_id": hit["entity"].get("doc_id", ""),
                "subject": hit["entity"].get("subject", ""),
                "grade": hit["entity"].get("grade", ""),
                "chapter": hit["entity"].get("chapter", ""),
                "knowledge_point": hit["entity"].get("knowledge_point", ""),
                "_source": "dense",
            })
        min_sim = settings.DENSE_MIN_SIMILARITY
        if min_sim > 0:
            before = len(docs)
            docs = [d for d in docs if d["score"] >= min_sim]
            logger.info(
                f"稠密检索相似度阈值: min={min_sim:.3f}, 保留 {len(docs)}/{before} 条"
            )
        return docs

    def _sparse_search(self, query: str, filter_str: str, top_k: int) -> list[dict]:
        """本地 BM25 稀疏检索"""
        if not self.bm25 or not self.bm25_docs:
            logger.debug("BM25 索引为空，跳过稀疏检索")
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # 排序并取 top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = self.bm25_docs[idx]
            # 如果有过滤条件，简易过滤
            if filter_str:
                if "subject" in filter_str and doc.get("subject", "") not in filter_str:
                    continue
                if "grade" in filter_str and doc.get("grade", "") not in filter_str:
                    continue
            results.append({
                "id": doc.get("id", 0),
                "text": doc.get("chunk_text", ""),
                "score": float(scores[idx]),
                "doc_id": doc.get("doc_id", ""),
                "subject": doc.get("subject", ""),
                "grade": doc.get("grade", ""),
                "chapter": doc.get("chapter", ""),
                "knowledge_point": doc.get("knowledge_point", ""),
                "_source": "sparse",
            })

        return results

    def _rrf_fusion(self, dense_results: list[dict], sparse_results: list[dict], top_k: int) -> list[dict]:
        """RRF（倒数排名融合）合并两路检索结果"""
        from collections import defaultdict

        k = settings.RRF_K
        score_map = defaultdict(float)

        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            score_map[doc_id] += 1.0 / (k + rank + 1)
            score_map[f"_{doc_id}_data"] = doc

        for rank, doc in enumerate(sparse_results):
            doc_id = doc["id"]
            score_map[doc_id] += 1.0 / (k + rank + 1)
            if f"_{doc_id}_data" not in score_map:
                score_map[f"_{doc_id}_data"] = doc

        # 按融合得分排序
        scored_docs = []
        for doc_id in score_map:
            if isinstance(doc_id, int) or (isinstance(doc_id, str) and doc_id.isdigit()):
                doc_id_num = int(doc_id)
                doc_data = score_map.get(f"_{doc_id_num}_data", {})
                if doc_data:
                    scored_docs.append((score_map[doc_id_num], doc_data))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # 重新分配归一化得分
        max_score = scored_docs[0][0] if scored_docs else 1
        results = []
        for score, doc in scored_docs[:top_k]:
            doc["score"] = round(score / max_score, 4)
            results.append(doc)

        return results

    def delete_by_doc_id(self, doc_id: str):
        """根据 doc_id 删除所有相关切片"""
        collection_name = settings.MILVUS_COLLECTION
        logger.info(f"删除 doc_id={doc_id} 的切片")
        self.milvus_client.delete(
            collection_name=collection_name,
            filter=f"doc_id == '{doc_id}'",
        )
        self._rebuild_bm25_index()
        logger.info(f"删除完成")

    @property
    def collection_stats(self) -> dict:
        """获取集合统计信息"""
        collection_name = settings.MILVUS_COLLECTION
        stats = self.milvus_client.get_collection_stats(collection_name)
        return {"row_count": stats.get("row_count", 0)}
