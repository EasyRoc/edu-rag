# 数据清洗功能测试文档

## 1. 功能概述

`IngestionPipeline._clean_file_docs` 是文件导入流程中的核心数据清洗环节，位于文档加载之后、切片入库之前。

清洗流水线由 `ingestion/cleaner.py` 的 `CleaningPipeline` 实现，处理链路为：

```
加载后的 Document → FileSourceAdapter.doc_to_records → CleaningPipeline.clean_batch → 清洗后的 Document
```

单条记录的清洗步骤：

```
Normalize → Denoise → Structure Repair → Dedup → Quality Filter → 输出 CleanRecord
```

| 步骤 | 组件 | 作用 |
|------|------|------|
| 1 | `Normalizer` | 去除不可见字符、规范化空格和换行 |
| 2 | `Denoiser` | 过滤页眉页脚、目录/版权声明、短文本、PDF 断词 |
| 3 | `StructureRepairer` | 修复断句、连字符拼接、Markdown 语法清洗 |
| 4 | 去重检查 | 基于 content hash 去除完全重复的内容 |
| 5 | `QualityFilter` | 评分过滤（长度 > 50、有结构、低噪声） |

## 2. 测试策略

### 2.1 单元测试（白盒）

针对每个清洗组件构造已知输入，断言输出是否符合预期。不依赖外部文件。

### 2.2 集成测试（灰盒）

- **文件级**：用 `sample_docs/` 下的真实文件走完整 `_clean_file_docs` 流程，验证端到端行为。
- **构造 Doc**：用 langchain Document 对象模拟特定脏数据场景，验证流水线能正确过滤/修复。

### 2.3 有效性验证方法

通过对比 **input count vs output count** + **CleanStats 各项指标** 判断清洗是否生效：

- `dedup_count > 0`：说明存在重复内容，去重生效
- `dropped_count > 0`：说明低质量/噪声内容被过滤
- `output_count < input_count`：整体数据量减少，清洗在起作用
- `output_count = input_count + dedup_count + dropped_count`：数据守恒成立
- 检查 `quality_score` 字段：清洗后的 Document 都有质量评分

## 3. 测试用例

### 3.1 Normalizer 测试

| 用例 | 输入 | 期望输出 |
|------|------|----------|
| 不可见字符清除 | `"hello\x00world"` | `"hello world"` → 不可见字符被移除 |
| 多余空格合并 | `"a    b   c"` | `"a b c"` |
| 多余换行合并 | `"a\n\n\n\nb"` | `"a\n\nb"`（最多保留 2 个连续换行） |
| 行尾空格清理 | `"abc   \n"` | `"abc"` |
| 空字符串 | `""` | `""`（返回空串，后续步骤会过滤） |

### 3.2 Denoiser 测试

| 用例 | 输入 | 期望输出 |
|------|------|----------|
| 纯页码 | `"123"` | `""`（被过滤） |
| 短文本（< 10 字符） | `"abc"` | `""`（被过滤） |
| 目录关键词 | `"目录"` | `""`（被过滤） |
| 版权声明 | `"2024 版权所有"` | `""`（被过滤） |
| 正常长文本 | `"这是一个包含足够多字符的有效文本内容"` | 原文本（保留） |
| PDF 断词修复 | `"hello-\nworld"` | `"helloworld"` |
| 页眉页码格式 | `"第1页"` | `""`（被过滤） |

### 3.3 StructureRepairer 测试

| 用例 | 输入 | 期望输出 |
|------|------|----------|
| PDF 断句合并 | `"这是第一句\n这是第二句"` | `"这是第一句这是第二句"`（单换行非段落边界） |
| Markdown 标题 | `"## 标题文本"` | `"标题文本"` |
| Markdown 加粗 | `"这是**粗体**文本"` | `"这是粗体文本"` |
| Markdown 链接 | `"[百度](https://baidu.com)"` | `"百度"` |
| Markdown 行内代码 | `` "foo`bar`baz" `` | `"foobarbaz"` |

### 3.4 QualityFilter 测试

| 用例 | 输入 | 期望结果 |
|------|------|----------|
| 长度 > 50 且有标点 | 大于 50 字符的有标点文本 | score >= 0.5，保留 |
| 长度 < 50 无标点 | `"abcdefg"` | score < 0.5，过滤 |
| 噪声比例高 | 大量控制字符的文本 | score < 0.5，过滤 |

### 3.5 去重测试

| 用例 | 输入 | 期望 |
|------|------|------|
| 两条完全相同内容 | 两个相同 content | 输出只有 1 条，dedup_count = 1 |
| 两条不同内容 | 两个不同 content | 输出 2 条，dedup_count = 0 |

### 3.6 集成测试（端到端）

| 用例 | 文件 | 验证点 |
|------|------|--------|
| PDF 清洗 | `sample_docs/教材_物理_浮力.pdf` | output_count <= input_count，content 无不可见字符，无页码残留 |
| Markdown 清洗 | `sample_docs/教材_数学_一元一次方程.md` | Markdown 符号被清除，标题纯文本化 |
| TXT 清洗 | `sample_docs/教材_语文_春.txt` | 换行规范化，空行合并 |
| 空内容模拟 | 构造全为短文本的 Documents | output_count = 0（全部被过滤） |

## 4. 运行测试

### 环境准备

```bash
cd edu-rag
pip install -r requirements.txt
```

### 执行测试

```bash
# 运行完整测试套件
python test/test_cleaner.py

# 仅运行单元测试（不依赖 sample_docs）
python test/test_cleaner.py --unit-only

# 仅运行集成测试
python test/test_cleaner.py --integration-only

# 输出详细日志
python test/test_cleaner.py --verbose
```

### 预期输出示例

```
============================================================
数据清洗功能测试套件
============================================================

[单元测试] Normalizer...
  ✓ 不可见字符清除
  ✓ 多余空格合并
  ✓ 多余换行合并
  ✓ 空字符串处理

[单元测试] Denoiser...
  ✓ 纯页码过滤
  ✓ 短文本过滤
  ✓ 目录关键词过滤
  ✓ PDF断词修复
  ...

[单元测试] StructureRepairer...
  ✓ PDF断句合并
  ✓ Markdown标题清洗
  ...

[单元测试] QualityFilter...
  ✓ 长文本保留
  ✓ 短文本过滤
  ...

[单元测试] 去重...
  ✓ 完全重复内容去重

[集成测试] 文件清洗...
  ✓ PDF 文件清洗 (教材_物理_浮力.pdf)
  ✓ Markdown 文件清洗 (教材_数学_一元一次方程.md)
  ✓ TXT 文件清洗 (教材_语文_春.txt)

============================================================
结果: 20/20 通过, 0 失败
============================================================
```

## 5. 测试结果解读

### 正常情况（清洗生效）

- `output_count < input_count`：部分脏数据被过滤
- `dedup_count >= 0`：去重计数
- `dropped_count > 0`（取决于输入质量）：低质量内容被丢弃
- 清洗后的 Document 都携带 `quality_score` 和 `content_hash`

### 异常情况需排查

- `output_count = 0`：所有数据被过滤，检查是 Normalizer 过度清洗还是 Denoiser 误杀
- `dedup_count 异常高`：数据源存在大量重复
- `quality_score 全为 0`：QualityFilter 阈值过高或输入质量极差

### 调参数指南

如需调整清洗强度，修改 `cleaner.py` 中的：
- `Normalizer._MULTI_NEWLINE`：控制最大连续换行数
- `Denoiser._HIGH_FREQ_THRESHOLD`：高频噪声阈值（默认 0.3）
- `QualityFilter.should_keep` 的 `min_score`：质量分阈值（默认 0.5）
- `Denoiser.denoise` 中的 `len(text.strip()) < 10`：最短文本长度

## 6. 自定义测试场景

在 `test/test_cleaner.py` 中添加新的测试方法即可，格式：

```python
def test_my_scenario(self):
    """我的测试场景描述"""
    # 构造输入
    cleaner = CleaningPipeline()
    result, stats = cleaner.clean_batch(my_records, source_type="pdf", source_id="test")
    # 断言
    assert stats.output_count == expected_value, "原因说明"
```
