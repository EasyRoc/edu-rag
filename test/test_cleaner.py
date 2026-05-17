"""数据清洗功能测试脚本

测试范围：ingestion.cleaner 模块全部组件 + IngestionPipeline._clean_file_docs 集成测试

用法:
    python test/test_cleaner.py                 # 全部测试
    python test/test_cleaner.py --unit-only     # 仅单元测试
    python test/test_cleaner.py --integration-only  # 仅集成测试
    python test/test_cleaner.py --verbose       # 详细输出
"""

import os
import sys
import argparse
from io import StringIO
from typing import Iterator

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document as LCDocument

from ingestion.cleaner import (
    Normalizer,
    Denoiser,
    StructureRepairer,
    QualityFilter,
    CleaningPipeline,
    CleanStats,
    CleanRecord,
    FileSourceAdapter,
    HashGenerator,
    IdGenerator,
    MetadataBuilder,
)


# ==================== 测试工具 ====================

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []

    def add(self, name: str, condition: bool, detail: str = ""):
        if condition:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            msg = f"  ✗ {name}" + (f" — {detail}" if detail else "")
            print(msg)
            self.failures.append(msg)

    def summary(self, title: str = "") -> str:
        total = self.passed + self.failed
        s = f"\n{'='*60}\n{title}结果: {self.passed}/{total} 通过"
        if self.failed:
            s += f", {self.failed} 失败\n"
            for f in self.failures:
                s += f"  {f}\n"
        else:
            s += " ✓"
        return s


def header(text: str):
    print(f"\n{'─'*50}\n[{text}]\n")


# ==================== 单元测试 ====================

def test_normalizer(result: TestResult, verbose: bool = False):
    header("单元测试: Normalizer")
    n = Normalizer()

    # 1. 不可见字符清除
    out = n.normalize("hello\x00\x08world")
    result.add("不可见字符清除", "\x00" not in out and "\x08" not in out, f"got: {repr(out)}")

    # 2. 多余空格合并
    out = n.normalize("a    b   c")
    result.add("多余空格合并", out == "a b c", f"got: {repr(out)}")

    # 3. 多余换行合并（>2 个连续换行 → 2 个）
    out = n.normalize("a\n\n\n\nb")
    result.add("多余换行合并", out == "a\n\nb", f"got: {repr(out)}")

    # 4. 行尾空格 + 尾部空白清除
    out = n.normalize("abc   \n")
    result.add("行尾空白清除", out == "abc", f"got: {repr(out)}")

    # 5. 空字符串
    out = n.normalize("")
    result.add("空字符串", out == "", f"got: {repr(out)}")

    # 6. 行尾空格后换行
    out = n.normalize("line1   \nline2")
    result.add("行尾空格+换行", out == "line1\nline2", f"got: {repr(out)}")

    # 7. 不可见 Unicode 字符（零宽空格等）
    out = n.normalize("text​‌more")
    result.add("零宽字符清除", out == "textmore", f"got: {repr(out)}")

    if verbose:
        print(f"  Normalizer 正则: INVISIBLE_CHARS={n._INVISIBLE_CHARS.pattern[:80]}...")


def test_denoiser(result: TestResult, verbose: bool = False):
    header("单元测试: Denoiser")
    d = Denoiser()

    # 1. 纯页码
    out = d.denoise("123", "pdf")
    result.add("纯页码过滤", out == "", f"got: {repr(out)}")

    # 2. 短文本（< 10 字符）
    out = d.denoise("abc", "txt")
    result.add("短文本过滤", out == "", f"got: {repr(out)}")

    # 3. 目录关键词
    out = d.denoise("目录", "pdf")
    result.add("目录关键词过滤", out == "", f"got: {repr(out)}")

    # 4. 版权声明
    out = d.denoise("2024 版权所有", "pdf")
    result.add("版权声明过滤", out == "", f"got: {repr(out)}")

    # 5. 正常长文本保留
    long_text = "这是一个包含足够多字符的有效文本内容，用于验证正常文本不会被过滤"
    out = d.denoise(long_text, "pdf")
    result.add("正常长文本保留", out == long_text.strip(), f"got: {repr(out[:30])}...")

    # 6. PDF 断词修复 (hello-\nworld → helloworld)
    out = d._denoise_pdf("hello-\nworld")
    result.add("PDF 连字符断词修复", out == "helloworld", f"got: {repr(out)}")

    # 7. 页眉页码 "第1页"
    out = d.denoise("第1页", "pdf")
    result.add("页码格式过滤(第X页)", out == "", f"got: {repr(out)}")

    # 8. 括号页码
    out = d.denoise("(123)", "pdf")
    result.add("括号页码过滤", out == "", f"got: {repr(out)}")

    # 9. 分页格式 "1/10"
    out = d.denoise("1/10", "pdf")
    result.add("分页格式过滤", out == "", f"got: {repr(out)}")

    # 10. 空格即为空
    out = d.denoise("     ", "txt")
    result.add("纯空格过滤", out == "", f"got: {repr(out)}")


def test_structure_repairer(result: TestResult, verbose: bool = False):
    header("单元测试: StructureRepairer")
    r = StructureRepairer()

    # 1. PDF 断句合并（非段落边界的单换行）
    out = r.repair("这是第一句\n这是第二句", "pdf")
    result.add("PDF断句合并", out == "这是第一句这是第二句", f"got: {repr(out)}")

    # 2. 保留段落边界（双换行）
    para_text = "段落一\n\n段落二"
    out = r.repair(para_text, "pdf")
    result.add("保留段落边界", "\n\n" in out, f"got: {repr(out)}")

    # 3. Markdown 标题清洗
    out = r.repair("## 标题文本", "md")
    result.add("Markdown标题清洗", out == "标题文本", f"got: {repr(out)}")

    # 4. Markdown 加粗清洗
    out = r.repair("这是**粗体**文本", "md")
    result.add("Markdown加粗清洗", out == "这是粗体文本", f"got: {repr(out)}")

    # 5. Markdown 链接清洗
    out = r.repair("[百度](https://baidu.com)", "md")
    result.add("Markdown链接清洗", out == "百度", f"got: {repr(out)}")

    # 6. Markdown 行内代码清洗
    out = r.repair("foo`bar`baz", "md")
    result.add("Markdown行内代码清洗", out == "foobarbaz", f"got: {repr(out)}")

    # 7. Markdown 斜体清洗
    out = r.repair("这是*斜体*文本", "md")
    result.add("Markdown斜体清洗", out == "这是斜体文本", f"got: {repr(out)}")

    # 8. 分隔线过滤
    out = r.repair("---", "md")
    result.add("分隔线过滤", out == "", f"got: {repr(out)}")

    # 9. TXT 断句处理
    out = r.repair("句子一\n句子二", "txt")
    result.add("TXT断句合并", out == "句子一句子二", f"got: {repr(out)}")


def test_quality_filter(result: TestResult, verbose: bool = False):
    header("单元测试: QualityFilter")
    qf = QualityFilter()

    # 1. 空内容得分为 0
    score = qf.score("")
    result.add("空内容得分为0", score == 0.0, f"got: {score}")

    # 2. 长文本 + 标点 → 高分
    long_good = "这是一个包含足够多字符、且具有标点符号的有效文本内容。" * 3
    score = qf.score(long_good)
    result.add("长文本+标点得分高", score >= 0.5, f"got: {score}")

    # 3. 短文本无标点 → 低分
    short_bad = "abcdefghij"
    score = qf.score(short_bad)
    result.add("短文本无标点得分低", score < 0.5, f"got: {score}")

    # 4. should_keep 判断
    result.add("长文本 should_keep=True",
               qf.should_keep(long_good) is True)
    result.add("短文本 should_keep=False",
               qf.should_keep(short_bad) is False)

    # 5. 含噪声文本——噪声比例 < 5%，应通过
    noisy = "\x00\x01\x02" + ("正常文本内容，包含足够的信息量和标点符号。" * 5)
    score = qf.score(noisy)
    result.add("噪声文本得分", score >= 0.5, f"got: {score}")
    if verbose:
        print(f"  noisy text score: {score}, len={len(noisy)}")


def test_dedup(result: TestResult, verbose: bool = False):
    header("单元测试: 去重")

    def make_records(texts: list[str]) -> Iterator[dict]:
        for t in texts:
            yield {"content": t, "position": "0", "page": 1}

    pipeline = CleaningPipeline()

    # 1. 完全重复内容（需 > 50 字符 + 标点，通过 QualityFilter）
    records = list(make_records([
        "这是一段足够长的重复内容，用于测试去重功能是否正常。需要超过五十个字符才能通过质量评分。",
        "这是一段足够长的重复内容，用于测试去重功能是否正常。需要超过五十个字符才能通过质量评分。",
        "这是另一段不同的内容文本，用于验证去重后不同内容能被正确保留下来，并且质量评分也能达标。",
    ]))
    results_list, stats = pipeline.clean_batch(iter(records), source_type="txt", source_id="test")
    result.add("重复内容去重(dedup_count=1)",
               stats.dedup_count == 1,
               f"dedup={stats.dedup_count}, output={stats.output_count}")
    result.add("去重后output_count正确",
               stats.output_count == 2,
               f"output={stats.output_count} (expect 2)")

    # 2. 数据守恒
    result.add("数据守恒(input=output+dedup+dropped)",
               stats.input_count == stats.output_count + stats.dedup_count + stats.dropped_count,
               f"in={stats.input_count} out={stats.output_count} "
               f"dedup={stats.dedup_count} drop={stats.dropped_count}")


def test_id_and_hash(result: TestResult, verbose: bool = False):
    header("单元测试: ID生成与Hash")

    # 1. 相同输入 → 相同 ID（幂等）
    id1 = IdGenerator.generate_readable("pdf", "test.pdf", "page_1")
    id2 = IdGenerator.generate_readable("pdf", "test.pdf", "page_1")
    result.add("ID幂等性", id1 == id2, f"{id1} vs {id2}")

    # 2. 不同 position → 不同 ID
    id3 = IdGenerator.generate_readable("pdf", "test.pdf", "page_2")
    result.add("不同position不同ID", id1 != id3, f"{id1} vs {id3}")

    # 3. Hash 幂等
    h1 = HashGenerator.generate("hello")
    h2 = HashGenerator.generate("hello")
    result.add("Hash幂等性", h1 == h2, f"{h1} vs {h2}")

    # 4. 不同内容不同 Hash
    h3 = HashGenerator.generate("world")
    result.add("不同内容不同Hash", h1 != h3, f"{h1} vs {h3}")


def test_file_source_adapter(result: TestResult, verbose: bool = False):
    header("单元测试: FileSourceAdapter")

    # 构造 langchain Documents
    docs = [
        LCDocument(page_content="第一页内容", metadata={"page": 1, "source_file": "test.pdf"}),
        LCDocument(page_content="第二页内容", metadata={"page": 2, "source_file": "test.pdf"}),
    ]
    records = list(FileSourceAdapter.doc_to_records(docs))
    result.add("Document转换数量正确", len(records) == 2, f"got: {len(records)}")
    result.add("content字段正确", records[0]["content"] == "第一页内容")
    result.add("position字段正确", records[0]["position"] == "page_1")
    result.add("page字段正确", records[0]["page"] == 1)
    result.add("extra传递原始metadata", records[0]["extra"]["source_file"] == "test.pdf")


def test_clean_stats(result: TestResult, verbose: bool = False):
    header("单元测试: CleanStats")

    stats = CleanStats(input_count=100, output_count=80, dedup_count=10, dropped_count=10)
    result.add("dedup_rate", stats.dedup_rate == 0.1, f"{stats.dedup_rate}")
    result.add("drop_rate", stats.drop_rate == 0.1, f"{stats.drop_rate}")

    zero_stats = CleanStats()
    result.add("空stats dedup_rate=0", zero_stats.dedup_rate == 0.0, f"{zero_stats.dedup_rate}")


def run_unit_tests(verbose: bool = False) -> TestResult:
    result = TestResult()
    test_normalizer(result, verbose)
    test_denoiser(result, verbose)
    test_structure_repairer(result, verbose)
    test_quality_filter(result, verbose)
    test_dedup(result, verbose)
    test_id_and_hash(result, verbose)
    test_file_source_adapter(result, verbose)
    test_clean_stats(result, verbose)
    return result


# ==================== 集成测试 ====================

def _clean_docs_via_pipeline(docs: list[LCDocument], source_type: str, file_path: str) -> tuple[list[LCDocument], CleanStats]:
    """模拟 _clean_file_docs 流程"""
    from ingestion.cleaner import CleaningPipeline, FileSourceAdapter

    pipeline = CleaningPipeline()
    source_id = os.path.basename(file_path)
    records = FileSourceAdapter.doc_to_records(docs)

    clean_results, stats = pipeline.clean_batch(
        records,
        source_type=source_type,
        source_id=source_id,
        file_name=source_id,
    )

    cleaned_docs = []
    for r in clean_results:
        cleaned_docs.append(
            LCDocument(
                page_content=r.content,
                metadata={
                    "source_file": source_id,
                    "file_type": source_type,
                    "clean_id": r.id,
                    "content_hash": r.metadata.get("content_hash", ""),
                    "quality_score": r.metadata.get("quality_score", 0.0),
                    "page": r.metadata.get("page", 0),
                },
            )
        )
    return cleaned_docs, stats


def test_integration_sample_docs(result: TestResult, verbose: bool = False):
    header("集成测试: sample_docs 文件清洗")

    from ingestion.loader import load_document

    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_docs")

    # PDF 和 TXT 使用真实文件；Markdown 使用预构造 Document 避免 UnstructuredMarkdownLoader 首次运行下载模型
    test_files = [
        ("教材_物理_浮力.pdf", "pdf", "loader"),
        ("教材_语文_春.txt", "txt", "loader"),
    ]

    for filename, expected_type, mode in test_files:
        filepath = os.path.join(sample_dir, filename)
        if not os.path.exists(filepath):
            result.add(f"文件存在: {filename}", False, "文件不存在，跳过")
            continue

        try:
            docs = load_document(filepath)
            input_count = len(docs)
            cleaned, stats = _clean_docs_via_pipeline(docs, expected_type, filepath)

            result.add(f"{filename}: output <= input",
                       stats.output_count <= stats.input_count,
                       f"in={stats.input_count} out={stats.output_count}")
            conservation = stats.input_count == stats.output_count + stats.dedup_count + stats.dropped_count
            result.add(f"{filename}: 数据守恒", conservation,
                       f"in={stats.input_count} out={stats.output_count} dedup={stats.dedup_count} drop={stats.dropped_count}")
            all_have_score = all("quality_score" in d.metadata for d in cleaned)
            result.add(f"{filename}: 清洗后Document含quality_score", all_have_score)
            has_invisible = any(
                any(ord(c) < 32 and c not in '\n\t' for c in doc.page_content)
                for doc in cleaned
            )
            result.add(f"{filename}: 无不可见字符残留", not has_invisible)

            if verbose:
                print(f"  {filename}: input={stats.input_count}, output={stats.output_count}, "
                      f"dedup={stats.dedup_count}, dropped={stats.dropped_count}, elapsed={stats.elapsed_ms}ms")
                if cleaned:
                    sample = cleaned[0]
                    content_preview = sample.page_content[:80].replace('\n', '\\n')
                    print(f"  样例: score={sample.metadata.get('quality_score')}, content={content_preview}...")
        except Exception as e:
            result.add(f"{filename}: 无异常", False, str(e))

    # Markdown：用预构造的 langchain Document 模拟加载结果，验证 Markdown 清洗逻辑
    md_docs = [
        LCDocument(page_content="## 一元一次方程\n\n**定义**：只含有一个未知数（元），未知数的次数都是1，等号两边都是整式。",
                   metadata={"page": 1, "source_file": "教材_数学_一元一次方程.md", "file_type": "md"}),
        LCDocument(page_content="### 等式的性质\n\n1. 等式两边加（或减）同一个数，结果仍相等。\n2. 等式两边乘同一个数，或除以同一个不为0的数，结果仍相等。",
                   metadata={"page": 2, "source_file": "教材_数学_一元一次方程.md", "file_type": "md"}),
        LCDocument(page_content="*重点*：去分母 → 去括号 → 移项 → 合并同类项 → 系数化为1",
                   metadata={"page": 3, "source_file": "教材_数学_一元一次方程.md", "file_type": "md"}),
    ]
    cleaned, stats = _clean_docs_via_pipeline(md_docs, "md", os.path.join(sample_dir, "教材_数学_一元一次方程.md"))
    result.add("Markdown预构造文档: output <= input",
               stats.output_count <= stats.input_count,
               f"in={stats.input_count} out={stats.output_count}")
    conservation = stats.input_count == stats.output_count + stats.dedup_count + stats.dropped_count
    result.add("Markdown预构造文档: 数据守恒", conservation,
               f"in={stats.input_count} out={stats.output_count} dedup={stats.dedup_count} drop={stats.dropped_count}")
    # 验证 Markdown 符号被清除
    has_md = any("##" in d.page_content or "**" in d.page_content or "*" in d.page_content.split('\n')[0]
                 for d in cleaned)
    result.add("Markdown预构造文档: Markdown符号已清除", not has_md)
    if verbose and cleaned:
        content_preview = cleaned[0].page_content[:80].replace('\n', '\\n')
        print(f"  Markdown预构造: input={stats.input_count}, output={stats.output_count}, "
              f"dedup={stats.dedup_count}, dropped={stats.dropped_count}, elapsed={stats.elapsed_ms}ms")
        print(f"  样例: score={cleaned[0].metadata.get('quality_score')}, content={content_preview}...")


def test_integration_dirty_docs(result: TestResult, verbose: bool = False):
    header("集成测试: 模拟脏数据")

    # 1. 全部短文本（应全部被过滤）
    dirty_docs = [
        LCDocument(page_content="123", metadata={"page": i, "source_file": "dirty.pdf"})
        for i in range(1, 6)
    ]
    cleaned, stats = _clean_docs_via_pipeline(dirty_docs, "pdf", "/tmp/dirty.pdf")
    result.add("全短文本全部过滤", stats.output_count == 0,
               f"output={stats.output_count} (expect 0)")

    # 2. 带不可见字符的文本
    dirty2 = [
        LCDocument(page_content="正常文本内容\x00夹杂不可见字符\x08测试" * 5,
                   metadata={"page": 1, "source_file": "dirty2.pdf"}),
    ]
    cleaned2, stats2 = _clean_docs_via_pipeline(dirty2, "pdf", "/tmp/dirty2.pdf")
    no_invisible = all(
        not any(ord(c) < 32 and c != '\n' for c in doc.page_content)
        for doc in cleaned2
    )
    result.add("不可见字符被清除", no_invisible, "checked cleaned docs")

    # 3. Markdown 符号清洗验证
    md_docs = [
        LCDocument(page_content="## 第一章 标题\n\n这是**重要**内容，请参考[文档](http://example.com)",
                   metadata={"page": 1, "source_file": "test.md"}),
    ]
    cleaned3, stats3 = _clean_docs_via_pipeline(md_docs, "md", "/tmp/test.md")
    if cleaned3:
        has_md_symbols = "##" in cleaned3[0].page_content or "**" in cleaned3[0].page_content
        result.add("Markdown符号被清除", not has_md_symbols,
                   f"content: {cleaned3[0].page_content[:80]}")


def run_integration_tests(verbose: bool = False) -> TestResult:
    result = TestResult()
    test_integration_sample_docs(result, verbose)
    test_integration_dirty_docs(result, verbose)
    return result


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(description="数据清洗功能测试套件")
    parser.add_argument("--unit-only", action="store_true", help="仅运行单元测试")
    parser.add_argument("--integration-only", action="store_true", help="仅运行集成测试")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()

    print("=" * 60)
    print("数据清洗功能测试套件")
    print("=" * 60)

    run_all = not args.unit_only and not args.integration_only

    unit_result = None
    integration_result = None

    if run_all or args.unit_only:
        unit_result = run_unit_tests(args.verbose)

    if run_all or args.integration_only:
        integration_result = run_integration_tests(args.verbose)

    # 汇总
    total_passed = 0
    total_failed = 0
    if unit_result:
        print(unit_result.summary("单元测试 "))
        total_passed += unit_result.passed
        total_failed += unit_result.failed
    if integration_result:
        print(integration_result.summary("集成测试 "))
        total_passed += integration_result.passed
        total_failed += integration_result.failed

    print(f"\n{'='*60}")
    print(f"总计: {total_passed}/{total_passed + total_failed} 通过", end="")
    if total_failed:
        print(f", {total_failed} 失败 ❌")
    else:
        print(" ✓")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
