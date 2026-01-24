#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计数据集中样本的 data[0] 长度分布，并输出分位数。

使用方式：
    python preprocess/stat_data_length.py /path/to/dataset

说明：
- dataset 目录下应包含 train/test/val 子目录
- 每个子目录下包含多个 .pkl 文件
- 每个 .pkl 文件为 list，元素为 dict，包含键 "data"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _is_array(obj) -> bool:
    if np is not None and isinstance(obj, np.ndarray):
        return True
    if torch is not None and isinstance(obj, torch.Tensor):
        return True
    return False


def _extract_first_length(data) -> Optional[int]:
    """
    取 data 的第一个元素的长度。
    - 若 data 是 (tokens, ...) 结构，返回 len(tokens)
    - 若 data 是 [sample1_data, sample2_data]，返回 len(sample1_data[0])
    """
    if isinstance(data, (list, tuple)):
        if not data:
            return None
        first = data[0]
        if isinstance(first, (list, tuple)):
            if not first:
                return None
            # 若 first[0] 仍是数组/张量，则取其长度
            if isinstance(first[0], (list, tuple)) or _is_array(first[0]):
                return len(first[0])
            # 否则 first 是 token 列表
            return len(first)
        if _is_array(first):
            return int(first.shape[0]) if hasattr(first, "shape") else len(first)
        return None
    if _is_array(data):
        return int(data.shape[0]) if hasattr(data, "shape") else len(data)
    return None


def _iter_pkl_files(split_dir: str) -> Iterable[str]:
    for filename in sorted(os.listdir(split_dir)):
        if filename.endswith(".pkl"):
            yield os.path.join(split_dir, filename)


def _collect_lengths(split_dir: str, verbose: bool = False) -> Tuple[Counter, int, int]:
    counts: Counter = Counter()
    total_samples = 0
    skipped_samples = 0

    for pkl_path in _iter_pkl_files(split_dir):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            print(f"⚠️  读取失败: {pkl_path} ({exc})")
            continue

        if not isinstance(data, list):
            print(f"⚠️  非列表数据，跳过: {pkl_path}")
            continue

        for idx, sample in enumerate(data):
            total_samples += 1
            if not isinstance(sample, dict) or "data" not in sample:
                skipped_samples += 1
                if verbose:
                    print(f"跳过样本 {pkl_path}:{idx}，缺少 data 字段")
                continue
            length = _extract_first_length(sample["data"])
            if length is None:
                skipped_samples += 1
                if verbose:
                    print(f"跳过样本 {pkl_path}:{idx}，无法解析长度")
                continue
            counts[length] += 1

    return counts, total_samples, skipped_samples


def _compute_percentiles(counts: Counter, percentiles: List[int]) -> Dict[int, int]:
    results: Dict[int, int] = {}
    total = sum(counts.values())
    if total == 0:
        return results

    targets = {p: math.ceil(p / 100.0 * total) for p in percentiles}
    sorted_targets = sorted(targets.items(), key=lambda x: x[1])

    cumulative = 0
    target_idx = 0
    for length, count in sorted(counts.items()):
        cumulative += count
        while target_idx < len(sorted_targets) and cumulative >= sorted_targets[target_idx][1]:
            percentile, _ = sorted_targets[target_idx]
            results[percentile] = length
            target_idx += 1
        if target_idx >= len(sorted_targets):
            break

    return results


def _print_summary(split_name: str, counts: Counter, total: int, skipped: int, percentiles: Dict[int, int]):
    valid = sum(counts.values())
    print("=" * 70)
    print(f"Split: {split_name}")
    print(f"样本总数: {total} | 有效: {valid} | 跳过: {skipped}")
    print("长度分布 (length -> count):")
    for length, count in sorted(counts.items()):
        print(f"  {length}: {count}")
    if percentiles:
        print("分位数长度:")
        for p in sorted(percentiles.keys()):
            print(f"  P{p}: {percentiles[p]}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 data[0] 长度分布与分位数")
    parser.add_argument("--dataset_path", help="数据集路径，包含 train/test/val 子目录")
    parser.add_argument(
        "--splits",
        default="train,test,val",
        help="要处理的子目录，逗号分隔 (默认: train,test,val)"
    )
    parser.add_argument(
        "--percentiles",
        default="50,55,60,65,70,75,80,85,90,95",
        help="分位数列表，逗号分隔 (默认: 50,55,...,95)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="结果 JSON 输出路径（默认: <dataset_path>/length_stats.json）"
    )
    parser.add_argument("--verbose", action="store_true", help="输出详细跳过信息")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = args.dataset_path

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    percentiles = [int(p) for p in args.percentiles.split(",") if p.strip()]

    output_path = args.output or os.path.join(dataset_path, "length_stats.json")

    overall_counts: Counter = Counter()
    overall_total = 0
    overall_skipped = 0
    processed_splits: List[str] = []
    missing_splits: List[str] = []

    for split in splits:
        split_dir = os.path.join(dataset_path, split)
        if not os.path.isdir(split_dir):
            print(f"⚠️  跳过不存在的子目录: {split_dir}")
            missing_splits.append(split)
            continue

        counts, total, skipped = _collect_lengths(split_dir, verbose=args.verbose)
        overall_counts.update(counts)
        overall_total += total
        overall_skipped += skipped
        processed_splits.append(split)

    percentile_values = _compute_percentiles(overall_counts, percentiles)
    _print_summary("all", overall_counts, overall_total, overall_skipped, percentile_values)

    result = {
        "dataset_path": dataset_path,
        "splits": processed_splits,
        "missing_splits": missing_splits,
        "total_samples": overall_total,
        "valid_samples": sum(overall_counts.values()),
        "skipped_samples": overall_skipped,
        "length_counts": {str(k): v for k, v in sorted(overall_counts.items())},
        "percentiles": {str(k): v for k, v in sorted(percentile_values.items())}
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ 结果已写入: {output_path}")


if __name__ == "__main__":
    main()
