"""数据清洗模块：Normalize → Denoise → Structure Repair → Validate

支持 PDF/MD/TXT/MySQL 多数据源，输出统一 CleanRecord 结构。
"""

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from utils.logger import logger


# ==================== 数据模型 ====================

@dataclass
class CleanRecord:
    """清洗后的统一数据结构"""
    id: str
    content: str
    metadata: dict[str, Any]


@dataclass
class CleanStats:
    """清洗过程统计信息"""
    input_count: int = 0
    output_count: int = 0
    dedup_count: int = 0
    dropped_count: int = 0
    elapsed_ms: float = 0.0

    @property
    def dedup_rate(self) -> float:
        return self.dedup_count / self.input_count if self.input_count > 0 else 0.0

    @property
    def drop_rate(self) -> float:
        return self.dropped_count / self.input_count if self.input_count > 0 else 0.0


# ==================== ID 生成器 ====================

class IdGenerator:
    """稳定 ID 生成器：同一数据 → ID 不变"""

    @staticmethod
    def generate(source: str, source_id: str, position: str = "") -> str:
        raw = f"{source}_{source_id}_{position}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def generate_readable(source: str, source_id: str, position: str = "") -> str:
        """生成可读 ID，如 mysql_product_123 / pdf_xxx_page_3"""
        short_hash = hashlib.md5(f"{source}_{source_id}_{position}".encode()).hexdigest()[:8]
        return f"{source}_{source_id}_{position}_{short_hash}"


# ==================== Hash 生成器 ====================

class HashGenerator:
    """内容 Hash 生成器：用于变更检测和去重"""

    @staticmethod
    def generate(content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()


# ==================== 清洗步骤 ====================

class Normalizer:
    """规范化：编码统一、空格/换行规范化"""

    _INVISIBLE_CHARS = re.compile(
        r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏ - ⁠-⁯﻿￰-￿]'
    )
    _MULTI_NEWLINE = re.compile(r'\n{3,}')
    _MULTI_SPACE = re.compile(r'[ \t]{2,}')
    _MULTI_BLANK = re.compile(r'[ \t]+\n')

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = self._INVISIBLE_CHARS.sub('', text)
        text = self._MULTI_SPACE.sub(' ', text)
        text = self._MULTI_BLANK.sub('\n', text)
        text = self._MULTI_NEWLINE.sub('\n\n', text)
        return text.strip()


class Denoiser:
    """去噪：页眉页脚、目录/版权声明、短文本、高频噪声"""

    # 常见页眉页脚模式
    _HEADER_FOOTER_PATTERNS = [
        re.compile(r'^\d{1,4}\s*$'),                          # 纯页码
        re.compile(r'^第[一二三四五六七八九十\d]+章\s*.*$'),      # 章节标题（保留）
        re.compile(r'^[\(（]?\d{1,4}[\)）]?\s*$'),              # 括号页码
        re.compile(r'^-\s*\d+\s*-$'),                          # - 123 -
        re.compile(r'^第[一二三四五六七八九十\d]+页\s*$'),      # 第X页
        re.compile(r'^\d+/\d+$'),                               # 1/10
        re.compile(r'^[©️®™]?\s*\d{4}.*(?:版权所有|All Rights Reserved)', re.IGNORECASE),
    ]

    # 目录/版权特征词
    _TOC_KEYWORDS = re.compile(
        r'^(目录|目\s*录|Contents|Table of Contents|版权声明|版权信息|免责声明|前言|序言|致谢)$',
        re.IGNORECASE,
    )

    # 高频噪声阈值：同一文本出现次数超过阈值则过滤
    _HIGH_FREQ_THRESHOLD = 0.3

    def __init__(self):
        self._freq_counter: dict[str, int] = {}
        self._total_blocks = 0

    def denoise(self, text: str, source_type: str = "") -> str:
        if not text or len(text.strip()) < 10:
            return ""

        # 目录/版权声明
        if self._TOC_KEYWORDS.match(text.strip()):
            return ""

        # PDF 特殊去噪
        if source_type == "pdf":
            text = self._denoise_pdf(text)

        return text.strip()

    def _denoise_pdf(self, text: str) -> str:
        """PDF 专项去噪"""
        for pattern in self._HEADER_FOOTER_PATTERNS:
            if pattern.match(text.strip()):
                return ""
        # 移除换行符导致的断词（PDF 常见问题）
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        return text

    def is_high_frequency(self, text: str) -> bool:
        """检测高频噪声"""
        h = hashlib.md5(text.encode()).hexdigest()
        self._freq_counter[h] = self._freq_counter.get(h, 0) + 1
        self._total_blocks += 1
        if self._total_blocks > 100:
            freq = self._freq_counter[h] / self._total_blocks
            return freq > self._HIGH_FREQ_THRESHOLD
        return False


class StructureRepairer:
    """结构修复：断句修复、连字符拼接、Markdown 处理、SQL 行转文本"""

    _SENTENCE_BREAK = re.compile(r'(?<=[^。！？\n])\n(?=[^\n])')
    _HYPHEN_BREAK = re.compile(r'(\w+)-\n(\w+)')
    _MD_LINK = re.compile(r'\[([^\]]*)\]\([^)]+\)')
    _MD_BOLD = re.compile(r'\*\*([^*]+)\*\*')
    _MD_ITALIC = re.compile(r'\*([^*]+)\*')
    _MD_CODE = re.compile(r'`([^`]+)`')
    _MD_HEADING = re.compile(r'^#{1,6}\s+', re.MULTILINE)
    _MD_LIST = re.compile(r'^[\s]*[-*+]\s+', re.MULTILINE)
    _MD_HR = re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE)

    def repair(self, text: str, source_type: str = "") -> str:
        if not text:
            return ""
        if source_type in ("pdf", "txt"):
            return self._repair_plain(text)
        elif source_type == "md":
            return self._repair_markdown(text)
        return text

    def _repair_plain(self, text: str) -> str:
        """PDF/TXT 断句修复和连字符拼接"""
        text = self._HYPHEN_BREAK.sub(r'\1\2', text)
        # 合并被换行打断的句子（非段落边界）
        text = self._SENTENCE_BREAK.sub('', text)
        return text

    def _repair_markdown(self, text: str) -> str:
        """Markdown：提取标题结构，去除 markdown 符号"""
        # 保留标题文本但去掉 # 符号
        text = self._MD_HEADING.sub('', text)
        text = self._MD_HR.sub('', text)
        text = self._MD_LIST.sub('', text)
        # 链接：保留文本
        text = self._MD_LINK.sub(r'\1', text)
        # 加粗/斜体：保留内容
        text = self._MD_BOLD.sub(r'\1', text)
        text = self._MD_ITALIC.sub(r'\1', text)
        # 行内代码：保留内容
        text = self._MD_CODE.sub(r'\1', text)
        return text

    @staticmethod
    def row_to_text(row: dict, field_map: dict[str, str] | None = None) -> str:
        """MySQL 行记录 → 语义文本"""
        if field_map:
            parts = [f"{label}:{row.get(field, '')}" for field, label in field_map.items()]
        else:
            parts = [f"{k}:{v}" for k, v in row.items()]
        return "，".join(parts)


class QualityFilter:
    """数据质量控制：评分 + 过滤"""

    @staticmethod
    def score(content: str) -> float:
        if not content:
            return 0.0
        s = 0.0
        if len(content) > 50:
            s += 0.3
        if QualityFilter._has_structure(content):
            s += 0.3
        if QualityFilter._low_noise(content):
            s += 0.4
        return round(s, 2)

    @staticmethod
    def _has_structure(content: str) -> bool:
        """检查是否有基本结构（标点、分段等）"""
        has_punct = bool(re.search(r'[。，、；：？！,.!?;:]', content))
        has_paragraphs = content.count('\n') >= 1
        return has_punct or has_paragraphs

    @staticmethod
    def _low_noise(content: str) -> bool:
        """检查噪声比例是否低"""
        noise_chars = len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', content))
        return noise_chars / max(len(content), 1) < 0.05

    @staticmethod
    def should_keep(content: str, min_score: float = 0.5) -> bool:
        return QualityFilter.score(content) >= min_score


# ==================== Metadata 构建器 ====================

class MetadataBuilder:
    """统一元数据构建"""

    @staticmethod
    def build(
        source_type: str,
        source_id: str = "",
        file_name: str = "",
        table_name: str = "",
        position: str = "",
        page: int = 0,
        tags: list[str] | None = None,
        **extra,
    ) -> dict:
        meta: dict[str, Any] = {
            "source": source_type,
            "source_id": source_id or file_name or table_name,
            "position": position,
        }
        if file_name:
            meta["file_name"] = file_name
        if table_name:
            meta["table_name"] = table_name
        if page > 0:
            meta["page"] = page
        if extra:
            meta.update(extra)
        if tags:
            meta["tags"] = tags
        return meta


# ==================== 清洗流水线 ====================

class CleaningPipeline:
    """数据清洗流水线：Normalize → Denoise → Structure Repair → Validate → Quality Filter"""

    def __init__(self):
        self.normalizer = Normalizer()
        self.denoiser = Denoiser()
        self.repairer = StructureRepairer()
        self.quality_filter = QualityFilter()
        self._seen_hashes: set[str] = set()

    def clean(
        self,
        content: str,
        source_type: str,
        source_id: str = "",
        file_name: str = "",
        position: str = "",
        page: int = 0,
        table_name: str = "",
        extra: dict | None = None,
    ) -> CleanRecord | None:
        """清洗单条记录，返回 CleanRecord 或 None（被过滤）"""
        # 1. Normalize
        text = self.normalizer.normalize(content)
        if not text:
            return None

        # 2. Denoise
        text = self.denoiser.denoise(text, source_type)
        if not text:
            return None

        # 3. Structure Repair
        text = self.repairer.repair(text, source_type)
        if not text:
            return None

        # 4. Generate ID & Hash
        doc_id = IdGenerator.generate_readable(source_type, source_id, position)
        content_hash = HashGenerator.generate(text)

        # 5. Dedup check
        if content_hash in self._seen_hashes:
            return None
        self._seen_hashes.add(content_hash)

        # 6. Quality filter
        if not self.quality_filter.should_keep(text):
            return None

        # 7. Build metadata
        meta = MetadataBuilder.build(
            source_type=source_type,
            source_id=source_id,
            file_name=file_name,
            table_name=table_name,
            position=position,
            page=page,
        )
        meta["content_hash"] = content_hash
        meta["quality_score"] = self.quality_filter.score(text)
        meta["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        if extra:
            meta.update(extra)

        return CleanRecord(id=doc_id, content=text, metadata=meta)

    def clean_batch(
        self,
        records: Iterator[dict],
        source_type: str,
        source_id: str = "",
        file_name: str = "",
    ) -> tuple[list[CleanRecord], CleanStats]:
        """批量清洗，返回 (清洗结果列表, 统计信息)"""
        stats = CleanStats()
        results: list[CleanRecord] = []
        start = time.time()

        for rec in records:
            stats.input_count += 1
            content = rec.get("content") or rec.get("text") or ""
            position = rec.get("position", "")
            page = rec.get("page", 0)

            h = HashGenerator.generate(content)
            if h in self._seen_hashes:
                stats.dedup_count += 1
                continue

            cleaned = self.clean(
                content=content,
                source_type=source_type,
                source_id=source_id,
                file_name=file_name,
                position=position,
                page=page,
                extra=rec.get("extra"),
            )
            if cleaned is None:
                stats.dropped_count += 1
            else:
                results.append(cleaned)
                stats.output_count += 1

        stats.elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.info(
            f"清洗完成: input={stats.input_count}, output={stats.output_count}, "
            f"dedup={stats.dedup_count}, dropped={stats.dropped_count}, "
            f"dedup_rate={stats.dedup_rate:.2%}, drop_rate={stats.drop_rate:.2%}, "
            f"elapsed={stats.elapsed_ms}ms"
        )
        return results, stats

    def reset_dedup(self):
        """重置去重缓存（每次新文件导入时调用）"""
        self._seen_hashes.clear()


# ==================== SQL 数据源适配器 ====================

class SQLSourceAdapter:
    """MySQL/PostgreSQL 数据源适配器：连接数据库 → 游标分页流式读取 → 行转文本

    使用 SQLAlchemy 连接数据库，通过主键游标分页（禁止 OFFSET）流式读取大表。
    """

    def __init__(
        self,
        db_url: str,
        table_name: str,
        field_map: dict[str, str] | None = None,
        id_column: str = "id",
        columns: list[str] | None = None,
        where_clause: str = "",
        batch_size: int = 1000,
    ):
        self.db_url = db_url
        self.table_name = table_name
        self.field_map = field_map
        self.id_column = id_column
        self.columns = columns
        self.where_clause = where_clause
        self.batch_size = batch_size

    def build_query(self, last_id: int = 0) -> str:
        """构建游标分页查询（禁止 OFFSET）"""
        if self.columns:
            cols = ", ".join(self.columns)
        else:
            cols = "*"

        sql = f"SELECT {cols} FROM {self.table_name}"
        conditions = []

        if last_id > 0:
            conditions.append(f"{self.id_column} > {last_id}")

        if self.where_clause:
            conditions.append(self.where_clause)

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += f" ORDER BY {self.id_column} LIMIT {self.batch_size}"
        return sql

    def stream_rows(self) -> Iterator[dict]:
        """流式读取数据库行，每批次 yield 一条记录（生成器，内存友好）"""
        from sqlalchemy import create_engine, text

        engine = create_engine(self.db_url, pool_pre_ping=True)
        try:
            with engine.connect() as conn:
                last_id = 0
                batch_count = 0
                while True:
                    sql = self.build_query(last_id)
                    logger.debug(f"执行查询: {sql}")
                    result = conn.execute(text(sql))
                    rows = result.fetchall()
                    if not rows:
                        break

                    for row in rows:
                        row_dict = dict(row._mapping)
                        row_pk = row_dict.get(self.id_column, last_id)
                        yield self._row_to_record(row_dict, row_pk)
                        last_id = row_pk

                    batch_count += 1
                    if len(rows) < self.batch_size:
                        break

                logger.info(f"SQL 流式读取完成: {batch_count} 批次, last_id={last_id}")
        finally:
            engine.dispose()

    def _row_to_record(self, row: dict, row_pk) -> dict:
        """将数据库行转为清洗输入记录"""
        text = StructureRepairer.row_to_text(row, self.field_map)
        return {
            "content": text,
            "position": f"row_{row_pk}",
            "extra": {
                "table_name": self.table_name,
                "row_id": str(row_pk),
            },
        }


# ==================== 文件源适配器 ====================

class FileSourceAdapter:
    """文件数据源适配器：将 langchain Document 转为清洗输入记录"""

    @staticmethod
    def doc_to_records(docs: list) -> Iterator[dict]:
        """将 langchain Document 列表转为清洗记录流"""
        for i, doc in enumerate(docs):
            yield {
                "content": doc.page_content,
                "position": f"page_{doc.metadata.get('page', i + 1)}",
                "page": doc.metadata.get("page", i + 1),
                "extra": doc.metadata,
            }
