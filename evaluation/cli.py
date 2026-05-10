#!/usr/bin/env python
"""RAGAS 评估 CLI —— 命令行运行离线评估

用法:
    # 基于历史 QA 记录评估
    python evaluation/cli.py --from-db --limit 50

    # 从测试文件评估
    python evaluation/cli.py --from-file evaluation/sample_test.json

    # 指定指标
    python evaluation/cli.py --from-db --metrics faithfulness,answer_relevancy

    # 保存结果
    python evaluation/cli.py --from-db --save
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from datasets import Dataset

from evaluation.dataset_builder import EvalDatasetBuilder
from evaluation.pipeline import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS 评估 CLI")
    parser.add_argument("--from-db", action="store_true", help="从 QA 历史记录构建数据集")
    parser.add_argument("--from-file", type=str, help="从 JSON/JSONL 文件加载数据集")
    parser.add_argument("--from-manual", nargs="+", help="手动传入问题（需配合 --answers --contexts）")
    parser.add_argument("--answers", nargs="+", help="手动评估的答案列表")
    parser.add_argument("--contexts", nargs="+", help="手动评估的上下文列表（每个问题一个字符串）")
    parser.add_argument("--limit", type=int, default=50, help="从 DB 提取的最大记录数")
    parser.add_argument("--subject", type=str, default=None, help="学科过滤")
    parser.add_argument("--metrics", type=str, default=None, help="评估指标，逗号分隔")
    parser.add_argument("--name", type=str, default="cli_eval", help="评估任务名称")
    parser.add_argument("--save", action="store_true", help="保存评估结果到数据库")
    parser.add_argument("--no-report", action="store_true", help="不输出报告")
    return parser.parse_args()


async def main():
    args = parse_args()

    # ---------- 构建 Dataset ----------
    dataset: Dataset | None = None
    if args.from_db:
        print("从 QA 历史记录构建数据集...")
        dataset = await EvalDatasetBuilder.from_db(limit=args.limit, subject=args.subject)
    elif args.from_file:
        print(f"从文件构建数据集: {args.from_file}")
        dataset = EvalDatasetBuilder.from_file(args.from_file)
    elif args.from_manual:
        if not args.answers or not args.contexts:
            print("错误: --from-manual 需配合 --answers 和 --contexts")
            sys.exit(1)
        dataset = EvalDatasetBuilder.from_manual(
            questions=args.from_manual,
            answers=args.answers,
            contexts_list=[[c] for c in args.contexts],
        )
    else:
        print("请指定数据源: --from-db, --from-file, 或 --from-manual")
        sys.exit(1)

    if dataset is None or len(dataset) == 0:
        print("数据集为空")
        sys.exit(1)

    print(f"数据集大小: {len(dataset)} 条样本")

    # ---------- 评估 ----------
    metrics = args.metrics.split(",") if args.metrics else None
    result = await run_evaluation(
        dataset=dataset,
        name=args.name,
        metrics=metrics,
        save_to_db=args.save,
    )

    # ---------- 结果 ----------
    report_shown = getattr(args, "no_report", False)
    if not report_shown:
        print("\n聚合得分:")
        for metric, score in sorted(result.scores.items()):
            print(f"  {metric:30s}: {score:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
