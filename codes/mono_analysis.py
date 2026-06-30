#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_alpha_effect.py
系统性地分析不同 alpha 取值下单调性约束相对于层级约束的增益与误伤。
遍历所有 BN 实例和样本量，基于 ground truth 量化以下指标：
  - mono_only_count : 单调性独有禁止边数
  - mono_only_true  : 其中真实边（误伤）
  - mono_only_noise : 其中噪声边（有效过滤）
  - precision       : mono_only_noise / mono_only_count
  - coverage        : mono_only_noise / total_noise_edges
"""

import json
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

# ===================== 配置 =====================
BASE_DIR = "../experiments/generated_bns"
# α 扫描范围：0.02 到 0.90，可根据需求调整
ALPHA_VALUES = [round(x, 2) for x in np.arange(0.02, 0.91, 0.02)]
# 需要排除一些极端 α？先全部包含。

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_gt_edges(gt_path):
    """从 ground truth JSON 提取真实有向边集合"""
    data = load_json(gt_path)
    edges = data.get('edges', {}).get('single_edges', [])
    return {(e['from'], e['to']) for e in edges}

def extract_hier_edges(hier_path):
    """从层级黑名单 JSON 提取禁止边集合"""
    data = load_json(hier_path)
    edges = data.get('forbidden_edges_by_hier', [])
    return {(e['from'], e['to']) for e in edges}

def get_all_nodes(gt_data):
    """从 ground truth JSON 获取所有节点名"""
    return list(gt_data.get('individual_variables', {}).keys())

def main():
    root = Path(BASE_DIR)
    all_records = []  # 存储每条统计记录

    # 遍历 BN 实例
    for bn_dir in sorted(root.glob("bn_*")):
        if not bn_dir.is_dir():
            continue
        bn_name = bn_dir.name

        # Ground truth 文件
        gt_file = bn_dir / "ground_truth" / "dag.json"
        if not gt_file.is_file():
            gt_file = bn_dir / "ground_truth" / "topo.json"
        if not gt_file.is_file():
            print(f"[跳过] {bn_name}: 无 ground truth 文件")
            continue
        gt_data = load_json(str(gt_file))
        gt_edges = extract_gt_edges(str(gt_file))
        all_nodes = get_all_nodes(gt_data)
        n_nodes = len(all_nodes)
        # 总可能的有向边（不含自环）
        total_possible_edges = n_nodes * (n_nodes - 1)
        total_noise_edges = total_possible_edges - len(gt_edges)

        # 层级黑名单
        hier_file = bn_dir / "ground_truth" / "topo_hier_blacklist.json"
        if not hier_file.is_file():
            print(f"[跳过] {bn_name}: 无层级黑名单文件")
            continue
        hier_set = extract_hier_edges(str(hier_file))

        # 遍历 data 下的样本量目录
        data_dir = bn_dir / "data"
        if not data_dir.is_dir():
            continue

        for sample_dir in sorted(data_dir.glob("S*")):
            if not sample_dir.is_dir() or not re.match(r'S\d+$', sample_dir.name, re.I):
                continue
            sample_name = sample_dir.name

            # 单调性 JSON 文件（只需一个，因为包含 all_pairs_pmono）
            mono_files = list(sample_dir.glob("*_mono_blacklist_alpha*.json"))
            if not mono_files:
                print(f"[跳过] {bn_name}/{sample_name}: 无单调性 JSON")
                continue
            mono_data = load_json(str(mono_files[0]))
            all_pairs = mono_data.get("all_pairs_pmono", [])
            if not all_pairs:
                print(f"[警告] {bn_name}/{sample_name}: all_pairs_pmono 为空")
                continue

            # 构建 P_mono 字典
            pmono_dict = {(pair['from'], pair['to']): pair['P_mono'] for pair in all_pairs}

            # 对每个 alpha 计算禁止边集合
            for alpha in ALPHA_VALUES:
                mono_set = set()
                for (u, v), p in pmono_dict.items():
                    if p < alpha:
                        mono_set.add((u, v))

                # 计算重叠指标
                intersection = hier_set & mono_set
                mono_only = mono_set - hier_set
                mono_only_true = mono_only & gt_edges
                mono_only_noise = mono_only - gt_edges

                record = {
                    'bn': bn_name,
                    'sample': sample_name,
                    'alpha': alpha,
                    'hier_size': len(hier_set),
                    'mono_size': len(mono_set),
                    'intersection': len(intersection),
                    'mono_only_count': len(mono_only),
                    'mono_only_true': len(mono_only_true),
                    'mono_only_noise': len(mono_only_noise),
                    'total_noise_edges': total_noise_edges
                }
                all_records.append(record)

    if not all_records:
        print("未收集到任何数据，请检查目录结构。")
        return

    # 转换为 DataFrame 并计算派生指标
    df = pd.DataFrame(all_records)
    df['precision'] = df['mono_only_noise'] / df['mono_only_count']
    df['coverage'] = df['mono_only_noise'] / df['total_noise_edges']
    # 处理极端情况：mono_only_count 为 0 时 precision 为 NaN，填充为 1.0（没有独有边时视为精确）
    df['precision'] = df['precision'].fillna(1.0)

    # 按 alpha 汇总（所有 BN 实例和样本量的平均）
    summary = df.groupby('alpha').agg(
        avg_mono_only=('mono_only_count', 'mean'),
        avg_mono_true=('mono_only_true', 'mean'),
        avg_mono_noise=('mono_only_noise', 'mean'),
        avg_precision=('precision', 'mean'),
        avg_coverage=('coverage', 'mean'),
        std_mono_only=('mono_only_count', 'std'),
        count=('mono_only_count', 'count')
    ).reset_index()

    # 输出结果
    print("\n=== 不同 alpha 下单调性约束增益与误伤汇总 (所有实例与样本量的均值) ===")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 120)
    print(summary.to_string(index=False))

    # 保存详细数据和汇总
    df.to_csv("alpha_analysis_detail.csv", index=False)
    summary.to_csv("alpha_analysis_summary.csv", index=False)
    print("\n详细数据已保存至 alpha_analysis_detail.csv")
    print("汇总数据已保存至 alpha_analysis_summary.csv")

if __name__ == "__main__":
    main()