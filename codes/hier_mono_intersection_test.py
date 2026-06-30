#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_constraint_overlap.py
分析层级约束与单调性约束产生的禁止边集合的重叠情况。
遍历所有 BN 实例和样本量，读取 ground truth、层级黑名单、单调性黑名单，
输出每个样本量下的重叠统计，并汇总。

运行时机：在 step01 生成完所有数据后即可运行。
"""

import json
import os
import re
from pathlib import Path


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_gt_edges(gt_json_path):
    """从 topo.json 提取真实有向边集合"""
    gt = load_json(gt_json_path)
    edges = gt.get('edges', {}).get('single_edges', [])
    return {(e['from'], e['to']) for e in edges}


def extract_hier_edges(hier_json_path):
    """从层级黑名单 JSON 提取禁止边集合"""
    data = load_json(hier_json_path)
    # 字段名为 forbidden_edges_by_hier
    edges = data.get('forbidden_edges_by_hier', [])
    return {(e['from'], e['to']) for e in edges}


def extract_mono_edges(mono_json_path, alpha=None):
    """
    从单调性 JSON 提取禁止边集合（P_mono < alpha）。
    如果 alpha 为 None，则使用 JSON 中 metadata.alpha 的值。
    """
    data = load_json(mono_json_path)
    if alpha is None:
        alpha = data.get('metadata', {}).get('alpha', 0.1)
    mono_set = set()
    if 'all_pairs_pmono' in data:
        for pair in data['all_pairs_pmono']:
            if pair['P_mono'] < alpha:
                mono_set.add((pair['from'], pair['to']))
    return mono_set


def analyze_single(gt_edges, hier_set, mono_set):
    """对单个样本进行分析，返回统计字典"""
    intersection = hier_set & mono_set
    mono_only = mono_set - hier_set
    hier_only = hier_set - mono_set

    # 在单调性独有边中，区分真实边和噪声边
    mono_only_true = mono_only & gt_edges
    mono_only_noise = mono_only - gt_edges

    total_possible_edges = len(gt_edges)  # 真实有向边数，用于参考

    return {
        'hier_total': len(hier_set),
        'mono_total': len(mono_set),
        'intersection': len(intersection),
        'mono_only_total': len(mono_only),
        'mono_only_true': len(mono_only_true),  # 误伤的真实边
        'mono_only_noise': len(mono_only_noise),  # 有效过滤的噪声边
        'hier_only': len(hier_only),
        'mono_only_true_list': list(mono_only_true),
        'mono_only_noise_list': list(mono_only_noise)[:10]  # 仅展示前10条避免过长
    }


def main():
    base_dir = "../experiments/generated_bns"
    root = Path(base_dir)

    # 存储所有统计结果
    all_stats = []

    for bn_dir in sorted(root.glob("bn_*")):
        if not bn_dir.is_dir():
            continue
        bn_name = bn_dir.name

        # Ground truth 边
        gt_file = bn_dir / "ground_truth" / "dag.json"  # 注意 ground truth 文件名
        if not gt_file.is_file():
            # 尝试 topo.json
            gt_file = bn_dir / "ground_truth" / "topo.json"
        if not gt_file.is_file():
            print(f"[警告] {bn_name} 缺少 ground truth 文件，跳过")
            continue
        gt_edges = extract_gt_edges(str(gt_file))

        # 层级黑名单
        hier_file = bn_dir / "ground_truth" / "topo_hier_blacklist.json"
        if not hier_file.is_file():
            print(f"[警告] {bn_name} 缺少层级黑名单文件，跳过")
            continue
        hier_set = extract_hier_edges(str(hier_file))

        # 遍历 data 下的每个样本量目录
        data_dir = bn_dir / "data"
        if not data_dir.is_dir():
            continue

        for sample_dir in sorted(data_dir.glob("S*")):
            if not sample_dir.is_dir() or not re.match(r'S\d+$', sample_dir.name, re.I):
                continue
            sample_name = sample_dir.name

            # 查找单调性黑名单文件
            mono_files = list(sample_dir.glob("*_mono_blacklist_alpha*.json"))
            if not mono_files:
                print(f"[警告] {bn_name}/{sample_name} 缺少单调性 JSON，跳过")
                continue
            mono_file = str(mono_files[0])

            # 提取单调性禁止边
            mono_set = extract_mono_edges(mono_file)

            # 分析
            stats = analyze_single(gt_edges, hier_set, mono_set)
            stats['bn_instance'] = bn_name
            stats['sample'] = sample_name
            all_stats.append(stats)

            # 打印详细结果
            print(f"\n=== {bn_name} / {sample_name} ===")
            print(f"层级禁止边总数: {stats['hier_total']}")
            print(f"单调性禁止边总数 (alpha={load_json(mono_file)['metadata']['alpha']}): {stats['mono_total']}")
            print(f"交集 (层级且单调性都禁止): {stats['intersection']}")
            print(f"单调性独有 (层级未禁止): {stats['mono_only_total']}")
            print(f"  其中真实边 (误伤): {stats['mono_only_true']} -> {stats['mono_only_true_list']}")
            print(f"  其中噪声边 (有效过滤): {stats['mono_only_noise']}")
            print(f"层级独有 (单调性未禁止): {stats['hier_only']}")
            if stats['mono_only_total'] == 0:
                print(">>> 结论：单调性禁止边完全是层级禁止边的子集，单调性在此样本下无额外过滤作用。")
            else:
                print(
                    f">>> 单调性独有边 {stats['mono_only_total']} 条，其中误伤 {stats['mono_only_true']} 条，有效过滤 {stats['mono_only_noise']} 条。")

    # 汇总摘要
    print("\n\n========== 汇总摘要 ==========")
    for stats in all_stats:
        print(f"{stats['bn_instance']}/{stats['sample']}: mono_all={stats['mono_total']}, mono_only={stats['mono_only_total']}, "
            f"误伤={stats['mono_only_true']}, 有效过滤={stats['mono_only_noise']}")

    # 统计所有样本中单调性独有边为0的比例
    zero_mono_only = sum(1 for s in all_stats if s['mono_only_total'] == 0)
    total_samples = len(all_stats)
    print(f"\n单调性独有边为0的样本数: {zero_mono_only}/{total_samples}")
    if zero_mono_only == total_samples:
        print("所有样本中单调性约束均无额外作用（单调性禁止边 ⊆ 层级禁止边）。")
    else:
        print("部分样本中单调性有额外过滤作用，请查看具体统计。")


if __name__ == "__main__":
    main()