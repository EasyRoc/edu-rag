#!/usr/bin/env python
"""RAGAS 评估 CLI —— 命令行运行离线评估 / 生成测试集 / 校验

用法:
    # 基于历史 QA 记录评估
    python evaluation/cli.py evaluate --from-db --limit 50

    # 从测试文件评估
    python evaluation/cli.py evaluate --from-file evaluation/sample_test.json

    # 指定指标
    python evaluation/cli.py evaluate --from-db --metrics faithfulness,answer_relevancy

    # 生成测试集（LLM 辅助）
    python evaluation/cli.py generate --subject math --grade junior --count 30

    # 校验测试集
    python evaluation/cli.py validate --file data/test_sets/manual_v1.jsonl

    # 从 QA 历史导出测试集
    python evaluation/cli.py export --min-feedback 1 --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from datasets import Dataset

from evaluation.dataset_builder import EvalDatasetBuilder
from evaluation.pipeline import run_evaluation
from evaluation.testset_generator import TestSetGenerator
from utils.logger import logger


# ===================================================================
# evaluate 子命令
# ===================================================================
def _add_evaluate_parser(subparsers) -> None:
    p = subparsers.add_parser("evaluate", help="运行离线评估")
    p.add_argument("--from-db", action="store_true", help="从 QA 历史记录构建数据集")
    p.add_argument("--from-file", type=str, help="从 JSON/JSONL 文件加载数据集")
    p.add_argument("--live", action="store_true", help="实时模式：先通过 RAG 系统生成回答，再评估（适用于只有 question 的测试集）")
    p.add_argument("--limit", type=int, default=50, help="从 DB 提取的最大记录数")
    p.add_argument("--subject", type=str, default=None, help="学科过滤")
    p.add_argument("--grade", type=str, default=None, help="年级过滤（live 模式）")
    p.add_argument("--metrics", type=str, default=None, help="评估指标，逗号分隔")
    p.add_argument("--name", type=str, default="cli_eval", help="评估任务名称")
    p.add_argument("--save", action="store_true", help="保存评估结果到数据库")
    p.set_defaults(func=_cmd_evaluate)


async def _cmd_evaluate(args: argparse.Namespace) -> None:
    from evaluation.pipeline import run_live_evaluation
    from main import _vector_store

    dataset: Dataset | None = None
    if args.from_db:
        print("从 QA 历史记录构建数据集...")
        dataset = await EvalDatasetBuilder.from_db(limit=args.limit, subject=args.subject)
    elif args.from_file and args.live:
        print(f"实时评估模式: {args.from_file}")
        items = EvalDatasetBuilder.from_file(args.from_file)
        questions = [items[i]["question"] for i in range(len(items))]
        ground_truths = [items[i].get("ground_truth", "") for i in range(len(items))]
        metrics = args.metrics.split(",") if args.metrics else None
        result = await run_live_evaluation(
            questions=questions,
            vector_store=_vector_store,
            subject=args.subject,
            grade=args.grade,
            metrics=metrics,
            name=args.name,
            ground_truths=ground_truths if any(ground_truths) else None,
        )
        print("\n聚合得分:")
        for metric, score in sorted(result.scores.items()):
            print(f"  {metric:30s}: {score:.4f}")
        return
    elif args.from_file:
        print(f"从文件构建数据集: {args.from_file}")
        dataset = EvalDatasetBuilder.from_file(args.from_file)
    else:
        print("请指定数据源: --from-db 或 --from-file")
        print("  测试只有 question 的测试集: --from-file data/test_sets/manual_v1.jsonl --live")
        sys.exit(1)

    if dataset is None or len(dataset) == 0:
        print("数据集为空")
        sys.exit(1)

    print(f"数据集大小: {len(dataset)} 条样本")

    metrics = args.metrics.split(",") if args.metrics else None
    result = await run_evaluation(
        dataset=dataset,
        name=args.name,
        metrics=metrics,
        save_to_db=args.save,
    )

    print("\n聚合得分:")
    for metric, score in sorted(result.scores.items()):
        print(f"  {metric:30s}: {score:.4f}")


# ===================================================================
# generate 子命令
# ===================================================================
def _add_generate_parser(subparsers) -> None:
    p = subparsers.add_parser("generate", help="从向量库文档用 LLM 生成测试集")
    p.add_argument("--subject", type=str, default=None, help="学科过滤")
    p.add_argument("--grade", type=str, default=None, help="年级段过滤（如 junior/senior/primary）")
    p.add_argument("--count", type=int, default=30, help="生成题目数量")
    p.add_argument("--output", type=str, default="data/test_sets/llm_gen.jsonl", help="输出 JSONL 路径")
    p.set_defaults(func=_cmd_generate)


async def _cmd_generate(args: argparse.Namespace) -> None:
    from main import _vector_store

    print(f"生成测试集: subject={args.subject}, grade={args.grade}, count={args.count}")
    generator = TestSetGenerator()
    items = await generator.from_vectorestore(
        vector_store=_vector_store,
        subject=args.subject,
        grade=args.grade,
        count=args.count,
    )

    if not items:
        print("未生成任何题目，请检查向量库是否有匹配的文档")
        sys.exit(1)

    # 校验
    report = TestSetGenerator.validate(items)
    print("\n测试集统计:")
    print(f"  生成: {report['total']} 题")
    print(f"  去重后: {report['total_after_dedup']} 题")
    print(f"  复杂度分布: {report['complexity_dist']}")
    print(f"  学科分布: {report['subject_dist']}")

    # 保存
    path = TestSetGenerator.save(items, args.output)
    print(f"\n已保存到: {path}")


# ===================================================================
# validate 子命令
# ===================================================================
def _add_validate_parser(subparsers) -> None:
    p = subparsers.add_parser("validate", help="校验测试集格式并输出统计信息")
    p.add_argument("--file", type=str, required=True, help="JSONL 文件路径")
    p.set_defaults(func=_cmd_validate)


def _cmd_validate(args: argparse.Namespace) -> None:
    path = Path(args.file)
    if not path.exists():
        print(f"文件不存在: {args.file}")
        sys.exit(1)

    items = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(json.loads(line))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]

    report = TestSetGenerator.validate(items)

    print(f"\n测试集校验报告: {args.file}")
    print(f"  原始条目: {report['total']}")
    print(f"  去重后: {report['total_after_dedup']}")
    print(f"  去重移除: {report['duplicates_removed']}")
    print(f"  缺失 question: {report['missing_question']}")
    print(f"  缺失 ground_truth: {report['missing_ground_truth']}")
    print(f"  缺失 contexts: {report['empty_contexts']}")
    print(f"  复杂度分布: {report['complexity_dist']}")
    print(f"  学科分布: {report['subject_dist']}")
    print(f"  年级分布: {report['grade_dist']}")
    print(f"  问题类型分布: {report['question_type_dist']}")


# ===================================================================
# export 子命令
# ===================================================================
def _add_export_parser(subparsers) -> None:
    p = subparsers.add_parser("export", help="从 QA 历史导出测试集（补全 ground_truth）")
    p.add_argument("--limit", type=int, default=50, help="最多导出条数")
    p.add_argument("--subject", type=str, default=None, help="学科过滤")
    p.add_argument("--min-feedback", type=int, default=None, help="最低 feedback 值（1=好评, -1=差评）")
    p.add_argument("--output", type=str, default="data/test_sets/from_qa.jsonl", help="输出 JSONL 路径")
    p.set_defaults(func=_cmd_export)


async def _cmd_export(args: argparse.Namespace) -> None:
    print(f"从 QA 历史导出测试集: limit={args.limit}, feedback>={args.min_feedback}")
    generator = TestSetGenerator()
    items = await generator.from_qa_history(
        limit=args.limit,
        subject=args.subject,
        feedback=args.min_feedback,
    )

    if not items:
        print("没有符合条件的 QA 记录")
        sys.exit(1)

    report = TestSetGenerator.validate(items)
    print(f"\n导出统计:")
    print(f"  导出: {len(items)} 条")
    print(f"  缺失 ground_truth: {report['missing_ground_truth']}")

    path = TestSetGenerator.save(items, args.output)
    print(f"\n已导出到: {path}")


# ===================================================================
# 主入口
# ===================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS 评估 / 测试集管理 CLI")
    subparsers = parser.add_subparsers(title="子命令", dest="command", help="可用命令")
    _add_evaluate_parser(subparsers)
    _add_generate_parser(subparsers)
    _add_validate_parser(subparsers)
    _add_export_parser(subparsers)
    return parser.parse_args()


async def main():
    args = parse_args()
    if not hasattr(args, "func"):
        print("请指定子命令: evaluate / generate / validate / export")
        print("示例: python evaluation/cli.py evaluate --from-db --limit 30")
        sys.exit(1)
    if asyncio.iscoroutinefunction(args.func):
        await args.func(args)
    else:
        args.func(args)


if __name__ == "__main__":
    asyncio.run(main())
