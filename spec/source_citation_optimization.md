# 📄 RAG 问答来源引用优化方案

---

## 1. 🎯 问题描述

当前页面问答返回的来源信息只有 **"📄 N 篇参考"** 字样，用户无法看到具体是哪个文档、哪一页、哪个章节。

### 现状截图（文字还原）

```
用户: 浮力的计算公式是什么？

知学助手: 浮力的计算公式为 F浮 = G排 = ρ液gV排 ......
           [1][2][3]

⏱ 3200ms  📊 一般  📄 3 篇参考  👍 👎
```

问题：**"📄 3 篇参考" 只是计数，不可点击，用户不知道这 3 篇参考来源是什么。**

---

## 2. 🔍 根因分析

| 环节 | 问题 | 位置 |
|------|------|------|
| **入库** | PDF 页码在切片时丢失，`chunker.py` 未将 `page` 写入 chunk 元数据 | `ingestion/chunker.py:60-69` |
| **入库** | Milvus schema 未存储 `source_file`、`page`、`file_type` 字段 | `core/vectorestore.py:43-59` |
| **检索** | 检索结果包含 `chapter`、`knowledge_point` 等元数据，但装配 references 时丢弃 | `services/rag_service.py:77-86` |
| **前端** | 参考面板 `#refs-panel` 存在但从未被填充数据 | `static/index.html:745-751` |
| **前端** | "📄 N 篇参考" 是纯文本，无点击事件 | `static/index.html:1180` |

---

## 3. 🏗️ 优化方案

### 总览

```
┌─ 入库层 ─────────────────────────────────────────────┐
│ chunker 保留 page + source_file → Milvus 动态字段存储  │
└──────────────────────────────────────────────────────┘
                          ↓
┌─ 检索/装配层 ────────────────────────────────────────┐
│ rag_service 装配完整元数据: source_file, page, chapter │
│ 新增 title 解析: doc_id → 查询 documents 表 → 文件名   │
└──────────────────────────────────────────────────────┘
                          ↓
┌─ 前端层 ────────────────────────────────────────────┐
│ ① "📄 N篇参考" → 可点击按钮，打开参考面板              │
│ ② 参考面板展示: 文件名、页码、相关章节、匹配分数        │
│ ③ 答案中 [1][2][3] 与参考列表序号对应                  │
└──────────────────────────────────────────────────────┘
```

---

## 4. 📋 分阶段实施计划

### Phase 1: 前端快速修复（低改动，立即可见效果）

**目标**：让用户至少能看到来源文件名。

**改动点**：

#### 4.1.1 rag_service.py — references 增加字段

```python
# services/rag_service.py:77-86
references = []
for doc in final_state.get("retrieved_docs", []):
    references.append({
        "index": len(references) + 1,       # 对应答案中的 [N]
        "text": doc.get("text", "")[:200],
        "source_file": doc.get("source_file", doc.get("doc_id", "")),  # 文件名
        "score": round(doc.get("score", 0), 4),
        "subject": doc.get("subject", ""),
        "chapter": doc.get("chapter", ""),           # 新增
    })
```

#### 4.1.2 index.html — 激活参考面板

改动点：

**① "📄 N 篇参考" 变为可点击按钮**（`static/index.html` 约 1180 行）:

```javascript
// 原来
${hasRefs ? `<span>📄 ${msg.references.length} 篇参考</span>` : ''}

// 改为
${hasRefs ? `<span class="ref-toggle" onclick="showRefs(${idx})">📄 ${msg.references.length} 篇参考 ▼</span>` : ''}
```

**② 实现 `showRefs(idx)` 函数**，填充 `#refs-list`:

```javascript
function showRefs(msgIdx) {
    const msg = chatMessages[msgIdx];
    if (!msg || !msg.references) return;
    
    const panel = document.getElementById('refs-panel');
    const list = document.getElementById('refs-list');
    const count = document.getElementById('refs-count');
    
    count.textContent = `📚 参考来源 (${msg.references.length})`;
    
    list.innerHTML = msg.references.map((ref, i) => `
        <div class="ref-item">
            <div class="ref-index">[${ref.index || i + 1}]</div>
            <div class="ref-body">
                <div class="ref-source">📁 ${escapeHtml(ref.source_file || '未知来源')}</div>
                ${ref.chapter ? `<div class="ref-chapter">📖 ${escapeHtml(ref.chapter)}</div>` : ''}
                <div class="ref-score">匹配度: ${(ref.score * 100).toFixed(0)}%</div>
                <div class="ref-text">${escapeHtml(ref.text).substring(0, 150)}...</div>
            </div>
        </div>
    `).join('');
    
    panel.classList.add('open');
}
```

**③ 参考面板样式增强**（在 `<style>` 中追加）:

```css
.ref-item {
    display: flex; gap: 10px; padding: 10px 12px;
    border-bottom: 1px solid #e8e8e8; cursor: default;
}
.ref-item:hover { background: #f5f7fa; }
.ref-index {
    font-weight: 700; color: #1677ff; font-size: 14px;
    min-width: 28px; flex-shrink: 0;
}
.ref-source { font-weight: 600; font-size: 13px; color: #333; }
.ref-chapter { font-size: 12px; color: #666; margin-top: 2px; }
.ref-score { font-size: 11px; color: #999; margin-top: 2px; }
.ref-text {
    font-size: 12px; color: #888; margin-top: 4px;
    overflow: hidden; text-overflow: ellipsis;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.ref-toggle { cursor: pointer; color: #1677ff; text-decoration: underline; }
.ref-toggle:hover { color: #0958d9; }
```

#### 4.1.3 前端数据传递 — 确保 `source_file` 在 retrieved_docs 中

当前 `_dense_search` 和 `_sparse_search` 返回的字段不包含 `source_file`。需要在 chunk 入库时将 `source_file` 写入 Milvus 动态字段，并在检索时返回。

**临时方案**（Phase 1）：从 `doc_id` 做简单的文件名映射。当前 `doc_id` 是直接使用 `source_file` 文件名作为 doc_id 而非 UUID（见 `vectorestore.py:92`：`"doc_id": meta.get("doc_id", ""),`），所以 Phase 1 可以将 `source` 字段的值改为更可读的形式。

---

### Phase 2: 入库链路修复（中期，根治数据缺失）

#### 4.2.1 chunker.py — 保留 page 和 source_file

```python
# ingestion/chunker.py split_documents() 中，每个 chunk 追加:
"page": doc.metadata.get("page", 0),
"source_file": doc.metadata.get("source_file", ""),
"file_type": doc.metadata.get("file_type", ""),
```

#### 4.2.2 vectorestore.py — insert_chunks 写入动态字段

Milvus schema 已开启 `enable_dynamic_field=True`，额外字段会自动存储。只需在 `insert_chunks` 的 `data` 列表中追加：

```python
data.append({
    ...
    "page": meta.get("page", 0),
    "source_file": meta.get("source_file", ""),
    "file_type": meta.get("file_type", ""),
})
```

#### 4.2.3 vectorestore.py — 检索返回 source_file 和 page

`hybrid_search` 的返回字典中增加：

```python
"page": hit.get("page", 0),
"source_file": hit.get("source_file", ""),
"file_type": hit.get("file_type", ""),
```

#### 4.2.4 rag_service.py — 装配完整元数据

```python
references.append({
    "index": len(references) + 1,
    "text": doc.get("text", "")[:200],
    "source_file": doc.get("source_file", "未知文档"),
    "page": doc.get("page", 0),
    "chapter": doc.get("chapter", ""),
    "score": round(doc.get("score", 0), 4),
    "subject": doc.get("subject", ""),
})
```

---

### Phase 3: 体验增强（长期，锦上添花）

| 功能 | 说明 | 优先级 |
|------|------|--------|
| **答案内引用高亮** | 答案中 `[1][2]` 可点击，点击后参考面板高亮对应条目 | P2 |
| **源文件预览** | 点击参考来源可弹窗预览原始文档上下文 | P2 |
| **引用跳转** | 点击参考条目后，页面滚动到答案中对应 `[N]` 位置 | P3 |
| **多文档聚合** | 同一来源文件的多个 chunk 合并显示 | P3 |
| **引用导出** | 支持导出带完整来源引用的问答记录 | P3 |

---

## 5. 📊 前后对比

### Before（当前）

```
⏱ 3200ms  📊 一般  📄 3 篇参考  👍 👎
                              ↑
                         不可点击，不知道是什么来源
```

### After（Phase 1 + 2 完成后）

```
⏱ 3200ms  📊 一般  📄 3 篇参考 ▼  👍 👎
                    ↑ 可点击打开面板

┌─ 📚 参考来源 (3) ───────────────────── ✕ ─┐
│                                            │
│ [1] 📁 教材_物理_浮力.pdf                   │
│     📖 第十章 浮力                          │
│     匹配度: 92%                             │
│     浮力是指浸在液体中的物体受到...            │
│                                            │
│ [2] 📁 教材_物理_浮力.pdf                   │
│     📖 10.2 阿基米德原理                     │
│     匹配度: 87%                             │
│     阿基米德原理：F浮 = G排 = ρ液gV排...     │
│                                            │
│ [3] 📁 教材_数学_一元一次方程.md              │
│     📖 等式的性质                            │
│     匹配度: 45%                             │
│     等式两边加同一个数，结果仍相等...          │
│                                            │
└────────────────────────────────────────────┘
```

---

## 6. 💾 改动文件清单

| 文件 | Phase | 改动类型 | 说明 |
|------|-------|----------|------|
| `static/index.html` | 1 | JS + CSS | 参考面板可点击、填充数据、样式 |
| `services/rag_service.py` | 1 | Python | references 增加字段 |
| `ingestion/chunker.py` | 2 | Python | chunk 元数据保留 page/source_file |
| `core/vectorestore.py` | 2 | Python | 存储 + 检索 source_file/page |
| `services/rag_service.py` | 2 | Python | 装配 page 等字段到 references |

---

## 7. ⚠️ 注意事项

1. **存量数据兼容**：Phase 2 修改入库链路后，已入库的旧 chunks 没有 `source_file` 和 `page` 动态字段。前端需处理空值（显示"未知来源"），或做一次全量 re-index。
2. **Milvus 动态字段**：已验证 `enable_dynamic_field=True` 已开启，无需 schema 变更，直接写入即可。
3. **性能影响**：references 新增字段不影响检索性能，只是前端多展示几个字段。
4. **向后兼容**：`source_file` 字段前端做空值兜底，`page=0` 时不显示页码行。
