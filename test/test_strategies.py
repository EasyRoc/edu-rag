"""多策略混合检索测试脚本

测试范围：core.strategies 模块全部组件 + core.nodes.retriever 集成测试

用法:
    python test/test_strategies.py                 # 全部测试
    python test/test_strategies.py --unit-only     # 仅单元测试
    python test/test_strategies.py --integration-only  # 仅集成测试
    python test/test_strategies.py --verbose       # 详细输出
"""

import os
import sys
import argparse
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings


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

def test_strategy_selection(result: TestResult, verbose: bool = False):
    header("单元测试: 策略选择器 select_strategy")
    from core.strategies.selector import StrategyType, select_strategy

    # U01: simple → DIRECT
    s = select_strategy("educational", "simple", "什么是浮力")
    result.add("U01: simple查询选DIRECT", s == StrategyType.DIRECT, f"got: {s}")

    # U02: medium → MULTI_QUERY
    s = select_strategy("educational", "medium", "浮力的应用")
    result.add("U02: medium查询选MULTI_QUERY", s == StrategyType.MULTI_QUERY, f"got: {s}")

    # U03: complex → DECOMPOSITION
    s = select_strategy("educational", "complex", "比较浮力和重力的区别并分析原理")
    result.add("U03: complex查询选DECOMPOSITION", s == StrategyType.DECOMPOSITION, f"got: {s}")

    # U04: 非教育意图一律 DIRECT
    for intent in ["chitchat", "greeting", "technical", "command", "other"]:
        s = select_strategy(intent, "complex", "test query")
        result.add(f"U04: {intent}意图选DIRECT", s == StrategyType.DIRECT, f"got: {s}")

    if verbose:
        print(f"  StrategyType values: {[s.value for s in StrategyType]}")


def test_quality_assessment(result: TestResult, verbose: bool = False):
    header("单元测试: 检索质量评估")
    from core.strategies.selector import assess_retrieval_quality

    # U05: 高质量结果通过
    high_docs = [
        {"score": 0.9, "text": "d1"},
        {"score": 0.8, "text": "d2"},
        {"score": 0.7, "text": "d3"},
    ]
    result.add("U05: 高质量结果通过", assess_retrieval_quality(high_docs) is True)

    # U06: 低平均分不通过
    low_docs = [{"score": 0.3, "text": "d1"}, {"score": 0.2, "text": "d2"}]
    result.add("U06: 低平均分不通过", assess_retrieval_quality(low_docs) is False)

    # U07: 结果数量不足
    few_docs = [{"score": 0.9, "text": "d1"}, {"score": 0.8, "text": "d2"}]
    result.add("U07: 结果不足3条不通过", assess_retrieval_quality(few_docs) is False)

    # U08: 空结果不通过
    result.add("U08: 空结果不通过", assess_retrieval_quality([]) is False)

    # 边界：刚好过线
    border_docs = [
        {"score": 0.5, "text": "d1"},
        {"score": 0.5, "text": "d2"},
        {"score": 0.5, "text": "d3"},
    ]
    result.add("边界: avg=0.5, count=3 通过", assess_retrieval_quality(border_docs) is True)


def test_hyde_step_back_triggers(result: TestResult, verbose: bool = False):
    header("单元测试: HyDE / Step-Back 触发条件")
    from core.strategies.selector import should_apply_hyde, should_apply_step_back

    # U09: top1 < 0.4 触发 HyDE
    result.add("U09: top1=0.3 触发HyDE",
               should_apply_hyde([{"score": 0.3}]) is True)
    result.add("U09: top1=0.2 触发HyDE",
               should_apply_hyde([{"score": 0.2, "text": "x"}, {"score": 0.1}]) is True)

    # U10: top1 >= 0.4 不触发 HyDE
    result.add("U10: top1=0.9 不触发HyDE",
               should_apply_hyde([{"score": 0.9}]) is False)
    result.add("U10: top1=0.4 不触发HyDE(边界)",
               should_apply_hyde([{"score": 0.4}]) is False)

    # 空结果触发
    result.add("空结果触发HyDE", should_apply_hyde([]) is True)
    result.add("空结果触发StepBack", should_apply_step_back([]) is True)

    # Step-Back 触发：少结果
    result.add("仅有1条结果触发StepBack",
               should_apply_step_back([{"score": 0.9}]) is True)
    # Step-Back 不触发：足够多且分高
    result.add("3条高分结果不触发StepBack",
               should_apply_step_back([
                   {"score": 0.9}, {"score": 0.8}, {"score": 0.7}
               ]) is False)

    if verbose:
        print(f"  HYDE_MIN_SCORE={getattr(settings, 'HYDE_MIN_SCORE', 0.4)}")
        print(f"  STEP_BACK_MIN_DOCS={getattr(settings, 'STEP_BACK_MIN_DOCS', 3)}")
        print(f"  RETRIEVAL_QUALITY_THRESHOLD={getattr(settings, 'RETRIEVAL_QUALITY_THRESHOLD', 0.5)}")


def test_multi_query_fusion(result: TestResult, verbose: bool = False):
    header("单元测试: 多查询RRF融合")
    from core.strategies.multi_query import multi_query_fusion

    # U11: RRF 融合排序正确
    r1 = [
        {"id": 1, "text": "chunk_a", "score": 0.9},
        {"id": 2, "text": "chunk_b", "score": 0.7},
    ]
    r2 = [
        {"id": 2, "text": "chunk_b", "score": 0.8},  # 重复
        {"id": 3, "text": "chunk_c", "score": 0.6},
    ]
    r3 = [
        {"id": 1, "text": "chunk_a", "score": 0.5},  # 重复
        {"id": 4, "text": "chunk_d", "score": 0.4},
    ]
    fused = multi_query_fusion([r1, r2, r3], top_k=5)
    result.add("U11: 4个唯一chunk全部返回", len(fused) == 4, f"got: {len(fused)}")
    # id=1 在 r1 rank0 和 r3 rank0 → RRF 最高
    result.add("U11: 重复chunk RRF加成后排名第一", fused[0]["id"] == 1, f"got: {fused[0]['id']}")
    # 分数归一化: 最高分应为 1.0
    result.add("U11: 最高分归一化为1.0", fused[0]["score"] == 1.0, f"got: {fused[0]['score']:.4f}")
    # 降序排列
    scores = [d["score"] for d in fused]
    result.add("U11: 按分数降序排列", scores == sorted(scores, reverse=True))

    # U12: 不修改原始数据
    result.add("U12: 原始r1[0] score不变", r1[0]["score"] == 0.9, f"got: {r1[0]['score']}")
    result.add("U12: 原始r2[1] score不变", r2[1]["score"] == 0.6, f"got: {r2[1]['score']}")

    # 空输入
    result.add("空输入返回空列表", multi_query_fusion([], top_k=5) == [])

    # 单路输入
    single = multi_query_fusion([r1], top_k=5)
    result.add("单路输入直接返回top_k", len(single) == min(5, len(r1)),
               f"got: {len(single)}")

    if verbose:
        for i, d in enumerate(fused):
            print(f"  fused[{i}]: id={d['id']}, score={d['score']:.4f}, text={d['text']}")


def test_merge_sub_results(result: TestResult, verbose: bool = False):
    header("单元测试: 子结果合并")
    from core.strategies.decomposition import merge_sub_results

    # U13: 保留最高分
    r1 = [
        {"id": 1, "score": 0.9, "text": "a"},
        {"id": 2, "score": 0.7, "text": "b"},
    ]
    r2 = [
        {"id": 2, "score": 0.8, "text": "b"},  # 同chunk更高分
        {"id": 3, "score": 0.6, "text": "c"},
    ]
    merged = merge_sub_results([r1, r2], top_k=5)
    result.add("U13: 去重后3条", len(merged) == 3, f"got: {len(merged)}")
    doc2 = [d for d in merged if d["id"] == 2][0]
    result.add("U13: 重复chunk保留最高分0.8", doc2["score"] == 0.8,
               f"got: {doc2['score']}")
    # 降序
    scores = [d["score"] for d in merged]
    result.add("U13: 按分数降序", scores == sorted(scores, reverse=True))

    # 截断 top_k
    many = [{"id": i, "score": 1.0 - i * 0.1} for i in range(10)]
    truncated = merge_sub_results([many], top_k=3)
    result.add("截断top_k=3", len(truncated) == 3, f"got: {len(truncated)}")

    # 空输入
    result.add("空输入返回空", merge_sub_results([], top_k=5) == [])


def test_variant_parsing(result: TestResult, verbose: bool = False):
    header("单元测试: LLM响应解析")
    from core.strategies.multi_query import generate_query_variants, MULTI_QUERY_SYSTEM

    # 验证 prompt 模板存在
    result.add("多查询system prompt非空", len(MULTI_QUERY_SYSTEM) > 0)

    from core.strategies.decomposition import DECOMPOSE_SYSTEM
    result.add("分解system prompt非空", len(DECOMPOSE_SYSTEM) > 0)

    from core.strategies.step_back import STEP_BACK_SYSTEM
    result.add("StepBack system prompt非空", len(STEP_BACK_SYSTEM) > 0)

    from core.strategies.hyde import HYDE_SYSTEM
    result.add("HyDE system prompt非空", len(HYDE_SYSTEM) > 0)

    # 验证函数签名可调用
    import inspect
    result.add("generate_query_variants是async函数",
               inspect.iscoroutinefunction(generate_query_variants))
    from core.strategies.decomposition import decompose_query
    result.add("decompose_query是async函数",
               inspect.iscoroutinefunction(decompose_query))
    from core.strategies.step_back import generate_step_back_query
    result.add("generate_step_back_query是async函数",
               inspect.iscoroutinefunction(generate_step_back_query))
    from core.strategies.hyde import generate_hypothetical_answer
    result.add("generate_hypothetical_answer是async函数",
               inspect.iscoroutinefunction(generate_hypothetical_answer))


def run_unit_tests(verbose: bool = False) -> TestResult:
    result = TestResult()
    test_strategy_selection(result, verbose)
    test_quality_assessment(result, verbose)
    test_hyde_step_back_triggers(result, verbose)
    test_multi_query_fusion(result, verbose)
    test_merge_sub_results(result, verbose)
    test_variant_parsing(result, verbose)
    return result


# ==================== 集成测试 ====================

# 共享 vector_store 实例，避免 Milvus Lite 多实例并发冲突
_shared_store = None


def _get_shared_store():
    """获取共享的 K12VectorStore 实例（懒加载，整个测试过程只创建一个）"""
    global _shared_store
    if _shared_store is None:
        try:
            from core.vectorestore import K12VectorStore
            _shared_store = K12VectorStore()
            stats = _shared_store.collection_stats
            count = stats.get("row_count", 0)
            if count > 0:
                print(f"  (Milvus 已连接，{count} 行数据)")
            else:
                print(f"  (Milvus 已连接，无数据)")
        except Exception as e:
            print(f"  (Milvus 不可用: {e})")
            _shared_store = None
    return _shared_store


def test_direct_retrieve_e2e(result: TestResult, verbose: bool = False):
    header("集成测试: DIRECT策略端到端 (I01)")
    store = _get_shared_store()
    if store is None:
        result.add("I01: DIRECT端到端", False, "Milvus不可用，跳过")
        return

    from core.nodes.retriever import _direct_retrieve

    docs = _direct_retrieve(store, "浮力", "simple")
    result.add("I01: simple查询返回结果", len(docs) > 0, f"got: {len(docs)} 条")
    result.add("I01: 结果含score字段", all("score" in d for d in docs))
    result.add("I01: 结果含text字段", all("text" in d for d in docs))
    result.add("I01: score在0-1范围", all(0 <= d["score"] <= 1 for d in docs))
    # simple 查询的 top_k=3
    result.add("I01: simple返回≤3条", len(docs) <= 3, f"got: {len(docs)}")

    # medium 查询
    docs2 = _direct_retrieve(store, "浮力的应用和原理", "medium")
    result.add("I01: medium查询返回≤5条", len(docs2) <= 5, f"got: {len(docs2)}")

    if verbose and docs:
        for i, d in enumerate(docs):
            print(f"  result[{i}]: score={d['score']:.4f}, text={d['text'][:60]}...")


def test_llm_unavailable_degradation(result: TestResult, verbose: bool = False):
    header("集成测试: LLM不可用时策略降级 (I02)")
    from core.strategies._llm import llm_complete

    # 测试 llm_complete 在无 API Key 时返回空字符串
    key_set = bool(settings.LLM_API_KEY)
    if key_set:
        print(f"  (LLM_API_KEY 已配置，测试降级行为需要 mock)")
        # 即使有 key，测试空 prompt 或直接验证降级逻辑
        # 模拟：如果 llm_complete 返回空字符串时的降级行为
        pass

    # 验证降级函数存在且可达
    from core.nodes.retriever import _direct_retrieve
    import inspect
    result.add("I02: _direct_retrieve可调用", callable(_direct_retrieve))
    result.add("I02: llm_complete可调用", callable(llm_complete))

    # 无 API Key 时，llm_complete 应返回空字符串
    if not key_set:
        async def _test_empty_llm():
            return await llm_complete("system", "user")
        resp = asyncio.run(_test_empty_llm())
        result.add("I02: 无API_KEY时llm_complete返回空", resp == "", f"got: {repr(resp)[:50]}")
    else:
        print("  (跳过空API_KEY测试，因为已配置)")


def test_retrieve_node_integration(result: TestResult, verbose: bool = False):
    header("集成测试: retrieve_node策略集成 (I03)")
    store = _get_shared_store()
    if store is None:
        result.add("I03: retrieve_node集成", False, "Milvus不可用，跳过")
        return

    from core.nodes.retriever import hybrid_retrieve

    async def _run():
        # simple 查询 → DIRECT 策略
        docs = await hybrid_retrieve(
            vector_store=store,
            query="什么是浮力",
            complexity="simple",
            intent="educational",
        )
        return docs

    docs = asyncio.run(_run())
    result.add("I03: simple查询不报错", True)
    result.add("I03: simple查询返回结果", len(docs) > 0, f"got: {len(docs)} 条")

    # medium 查询 → MULTI_QUERY 策略（LLM 不可用时降级为 DIRECT）
    async def _run_medium():
        return await hybrid_retrieve(
            vector_store=store,
            query="浮力的应用和原理",
            complexity="medium",
            intent="educational",
        )

    docs2 = asyncio.run(_run_medium())
    result.add("I03: medium查询不报错（降级或正常）", True)
    result.add("I03: medium查询返回结果", len(docs2) > 0, f"got: {len(docs2)} 条")

    # complex 查询 → DECOMPOSITION 策略（LLM 不可用时降级为 DIRECT）
    async def _run_complex():
        return await hybrid_retrieve(
            vector_store=store,
            query="比较浮力和重力的区别并分析阿基米德原理",
            complexity="complex",
            intent="educational",
        )

    docs3 = asyncio.run(_run_complex())
    result.add("I03: complex查询不报错（降级或正常）", True)
    result.add("I03: complex查询返回结果", len(docs3) > 0, f"got: {len(docs3)} 条")

    if verbose:
        print(f"  simple results: {len(docs)}, medium: {len(docs2)}, complex: {len(docs3)}")


def test_top_k_mapping(result: TestResult, verbose: bool = False):
    header("集成测试: top_k映射")
    from core.nodes.retriever import _top_k_for

    result.add("simple → 3", _top_k_for("simple") == 3)
    result.add("medium → 5", _top_k_for("medium") == 5)
    result.add("complex → 8", _top_k_for("complex") == 8)
    result.add("unknown → 5 (default)", _top_k_for("unknown") == 5)


def test_config_values(result: TestResult, verbose: bool = False):
    header("集成测试: 配置项读取")
    result.add("MULTI_QUERY_VARIANTS存在", hasattr(settings, 'MULTI_QUERY_VARIANTS'))
    result.add("DECOMPOSITION_MAX_SUB存在", hasattr(settings, 'DECOMPOSITION_MAX_SUB'))
    result.add("RETRIEVAL_QUALITY_THRESHOLD存在", hasattr(settings, 'RETRIEVAL_QUALITY_THRESHOLD'))
    result.add("HYDE_MIN_SCORE存在", hasattr(settings, 'HYDE_MIN_SCORE'))
    result.add("STEP_BACK_MIN_DOCS存在", hasattr(settings, 'STEP_BACK_MIN_DOCS'))

    if verbose:
        print(f"  MULTI_QUERY_VARIANTS={settings.MULTI_QUERY_VARIANTS}")
        print(f"  DECOMPOSITION_MAX_SUB={settings.DECOMPOSITION_MAX_SUB}")
        print(f"  RETRIEVAL_QUALITY_THRESHOLD={settings.RETRIEVAL_QUALITY_THRESHOLD}")
        print(f"  HYDE_MIN_SCORE={settings.HYDE_MIN_SCORE}")
        print(f"  STEP_BACK_MIN_DOCS={settings.STEP_BACK_MIN_DOCS}")


def run_integration_tests(verbose: bool = False) -> TestResult:
    result = TestResult()
    test_config_values(result, verbose)
    test_top_k_mapping(result, verbose)
    test_direct_retrieve_e2e(result, verbose)
    test_llm_unavailable_degradation(result, verbose)
    test_retrieve_node_integration(result, verbose)
    return result


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(description="多策略混合检索测试套件")
    parser.add_argument("--unit-only", action="store_true", help="仅运行单元测试")
    parser.add_argument("--integration-only", action="store_true", help="仅运行集成测试")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()

    print("=" * 60)
    print("多策略混合检索 测试套件")
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
        print(f", {total_failed} 失败")
    else:
        print(" ✓")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
