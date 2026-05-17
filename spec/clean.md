

---

# 🧠 RAG 数据清洗模块 Spec（v3 · 可增删改版）

---

# 1. 🎯 目标（Objective）

构建一个支持多数据源的数据清洗系统，输出**可追踪、可更新、可删除**的数据结构，为后续向量索引和增量同步提供基础。

---

# 2. 📦 范围（Scope）

---

## ✅ 包含

* 数据清洗（去噪 / 标准化 / 结构修复）
* 流式处理（大文件 / 大数据量）
* 唯一ID生成（稳定ID）
* 内容Hash计算（变更检测）
* 元数据构建（支持追踪）

---

## ❌ 不包含

* chunk切分
* embedding
* 向量库操作（upsert/delete）

---

# 3. 🏗️ 架构设计（Architecture）

```text
Data Source
    ↓
Stream Reader ⭐（流式读取）
    ↓
Source Adapter
    ↓
Cleaning Pipeline ⭐
    ↓
ID Generator ⭐（稳定ID）
    ↓
Metadata Builder ⭐
    ↓
Quality Filter
    ↓
Clean Output（统一结构）
```

---

# 4. 📑 输入输出规范（Data Contract）

---

## 4.1 输入

```json
{
  "source_type": "pdf | md | txt | mysql",
  "data": "...",
  "extra": {
    "file_name": "...",
    "file_path": "...",
    "table_name": "...",
    "row_id": "...",
    "page": 1,
    "offset": 0
  }
}
```

---

## 4.2 输出（核心结构）

```json
{
  "id": "唯一稳定ID",
  "content": "清洗后的文本",
  "metadata": {
    "source": "pdf/mysql/md/txt",
    "source_id": "原始数据主键或文件标识",
    "file_name": "...",
    "table_name": "...",
    "position": "page_1 / row_123",
    "content_hash": "md5值",
    "timestamp": "...",
    "tags": [],
    "quality_score": 0.0
  }
}
```

---

# 5. ⚙️ 核心能力设计

---

# 5.1 流式处理（Streaming）⭐必须

---

## PDF / TXT / MD

```python
def stream_file(file):
    for block in file:
        yield block
```

---

## MySQL（关键）

```sql
SELECT * FROM table
WHERE id > last_id
ORDER BY id
LIMIT 1000
```

---

👉 禁止：

```sql
LIMIT 1000000 OFFSET 0 ❌
```

---

---

# 5.2 清洗流水线（Cleaning Pipeline）

---

## Pipeline

```text
Normalize → Denoise → Structure Repair → Validate
```

---

## Normalize

* 编码统一
* 空格/换行规范化

---

## Denoise

* PDF页眉页脚删除
* 删除“目录 / 版权声明”
* 删除短文本

---

## Structure Repair

---

### PDF/TXT

* 断句修复
* 连字符拼接

---

### Markdown

* 提取标题结构
* 去 markdown 符号

---

### MySQL

👉 结构 → 语义文本

```python
def row_to_text(row):
    return f"商品:{row['name']}，价格:{row['price']}，销量:{row['sales']}"
```

---

---

# 5.3 稳定 ID 设计（核心⭐）

---

## ❗ 原则

```text
同一数据 → ID不变
数据变化 → ID可定位
```

---

## 设计公式

```python
id = hash(source + source_id + position)
```

---

## 示例

```text
mysql_product_123
pdf_xxx_page_3_block_2
md_doc_section_1
```

---

---

# 5.4 content_hash（变更检测）⭐必须

---

## 生成

```python
import hashlib

def get_hash(content):
    return hashlib.md5(content.encode()).hexdigest()
```

---

## 用途

| 场景   | 作用        |
| ---- | --------- |
| 更新判断 | hash变了才更新 |
| 去重   | 相同内容过滤    |

---

---

# 5.5 Metadata 设计（增强版）

---

## 必须字段

| 字段           | 说明      |
| ------------ | ------- |
| source       | 数据来源    |
| source_id    | 主键/文件ID |
| position     | 定位信息    |
| content_hash | 内容hash  |

---

## 可选字段

* page（PDF）
* table_name（MySQL）
* section（MD）

---

---

# 5.6 大文件处理（专项设计）

---

## ❗ 问题

* 噪声放大
* 内存爆炸
* 语义污染

---

## ✅ 方案

---

### 1️⃣ 分段处理

```text
文件 → 页 → 段落
```

---

---

### 2️⃣ 去重

```python
if hash(content) in seen:
    drop
```

---

---

### 3️⃣ 高频噪声过滤

```python
if freq(text) > threshold:
    drop
```

---

---

# 5.7 MySQL 大数据量处理

---

## 核心策略

---

### 1️⃣ 增量扫描

```sql
WHERE update_time > last_sync_time
```

---

---

### 2️⃣ 主键分页

```sql
WHERE id > last_id
LIMIT 1000
```

---

---

### 3️⃣ 字段裁剪

```sql
SELECT name, price, sales FROM table
```

---

---

# 5.8 数据质量控制

---

## 评分

```python
score = 0
if len(content) > 50: score += 0.3
if has_structure(content): score += 0.3
if low_noise(content): score += 0.4
```

---

## 过滤

```python
if score < 0.5:
    drop
```

---

---

# 6. 🧩 模块设计（Python）

---

## 核心接口

```python
class Cleaner:
    def clean(self, data) -> dict:
        pass
```

---

## ID生成器

```python
class IdGenerator:
    def generate(self, source, source_id, position):
        return hash(f"{source}_{source_id}_{position}")
```

---

---

## Hash生成器

```python
class HashGenerator:
    def generate(self, content):
        return md5(content)
```

---

---

## Pipeline

```python
def process(data):
    content = clean(data)
    doc_id = generate_id(...)
    content_hash = hash(content)

    return {
        "id": doc_id,
        "content": content,
        "metadata": {
            ...
        }
    }
```

---

---

# 7. 📊 可观测性（Observability）

---

必须记录：

* 输入数据量
* 输出数据量
* 去重率
* 丢弃率
* 平均处理耗时

---

---

# 8. 🚀 扩展能力

---

* LLM清洗（语义优化）
* 自动标签生成
* 表格解析（PDF）

---

---
