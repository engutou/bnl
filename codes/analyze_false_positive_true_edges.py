#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_false_positive_true_edges.py
分析被单调性约束误伤的真实边的特征：
  1. 它们的 P_mono 值分布（与 α 的差距）
  2. 它们的 BDeu 得分在所有候选父节点中的排名
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from pgmpy.estimators import BDeu

# ===================== 路径配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
BASE_DIR = os.path.join(EXPERIMENTS_DIR, "generated_bns")
OUTPUT_DIR = os.path.join(EXPERIMENTS_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 参数 =====================
SAMPLE_SIZES = ['S100', 'S200', 'S500', 'S1000', 'S5000', 'S10000']
ALPHA_VALUES = [0.4, 0.5, 0.6, 0.7]
ESS = 1.0  # BDeu 等效样本大小


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_bdeu_ranks(data, gt_edges, ess=1.0):
    """
    计算每条真实边的 BDeu 得分在其子节点所有候选父节点中的排名。
    返回字典 {(u, v): rank_percentile}
    """
    scorer = BDeu(data, equivalent_sample_size=ess)
    ranks = {}

    # 构建每个子节点的父节点候选列表
    nodes = list(data.columns)
    for u, v in gt_edges:
        if u not in nodes or v not in nodes:
            continue
        # 计算所有可能的父节点（除自己外）的 BDeu 增益
        scores = {}
        for candidate in nodes:
            if candidate == v:
                continue
            # 计算只有该候选父节点时的局部得分
            base_score = scorer.local_score(v, [])
            candidate_score = scorer.local_score(v, [candidate])
            gain = candidate_score - base_score
            scores[candidate] = gain

        # 真实父节点 u 的增益
        true_gain = scores.get(u, -np.inf)
        # 排名（增益越高越好）
        sorted_gains = sorted(scores.values(), reverse=True)
        rank = sorted_gains.index(true_gain) + 1 if true_gain in sorted_gains else len(sorted_gains)
        percentile = rank / len(sorted_gains)  # 1.0 = 最高, 0.0 = 最低
        ranks[(u, v)] = percentile
    return ranks


def main():
    root = Path(BASE_DIR)
    records = []

    for bn_dir in sorted(root.glob("bn_*")):
        if not bn_dir.is_dir():
            continue
        bn_name = bn_dir.name

        # Ground Truth
        gt_file = bn_dir / "ground_truth" / "dag.json"
        if not gt_file.is_file():
            gt_file = bn_dir / "ground_truth" / "topo.json"
        if not gt_file.is_file():
            continue
        gt_data = load_json(str(gt_file))
        gt_edges = [(e['from'], e['to']) for e in gt_data['edges']['single_edges']]
        true_edge_set = set(gt_edges)

        # 遍历样本量
        for sample_name in SAMPLE_SIZES:
            sample_dir = bn_dir / "data" / sample_name
            csv_files = list(sample_dir.glob("*.csv"))
            mono_files = list(sample_dir.glob("*_mono_blacklist_alpha*.json"))
            if not csv_files or not mono_files:
                continue

            data = pd.read_csv(str(csv_files[0]), sep=',')
            mono_data = load_json(str(mono_files[0]))
            all_pairs = mono_data.get('all_pairs_pmono', [])

            # 构建 P_mono 字典
            pmono_dict = {(p['from'], p['to']): p['P_mono'] for p in all_pairs}

            # 计算 BDeu 排名（一次性，只依赖数据和真图）
            bdeu_ranks = compute_bdeu_ranks(data, gt_edges, ess=ESS)

            # 对每个 alpha 分析被误伤的真实边
            for alpha in ALPHA_VALUES:
                # 找出被误伤的真实边
                false_positives = []
                for u, v in gt_edges:
                    p = pmono_dict.get((u, v), 1.0)
                    if p < alpha:
                        false_positives.append((u, v, p, bdeu_ranks.get((u, v), np.nan)))

                if false_positives:
                    fp_pmono = [x[2] for x in false_positives]
                    fp_ranks = [x[3] for x in false_positives if not np.isnan(x[3])]
                    records.append({
                        'bn': bn_name,
                        'sample': sample_name,
                        'alpha': alpha,
                        'fp_count': len(false_positives),
                        'fp_pmono_mean': np.mean(fp_pmono),
                        'fp_pmono_min': np.min(fp_pmono),
                        'fp_pmono_gap': alpha - np.mean(fp_pmono),  # α 与平均 P_mono 的差距
                        'fp_bdeu_rank_mean': np.mean(fp_ranks) if fp_ranks else np.nan,
                        'fp_bdeu_rank_median': np.median(fp_ranks) if fp_ranks else np.nan,
                    })
                else:
                    records.append({
                        'bn': bn_name,
                        'sample': sample_name,
                        'alpha': alpha,
                        'fp_count': 0,
                        'fp_pmono_mean': np.nan,
                        'fp_pmono_min': np.nan,
                        'fp_pmono_gap': np.nan,
                        'fp_bdeu_rank_mean': np.nan,
                        'fp_bdeu_rank_median': np.nan,
                    })

    # ===================== 汇总与输出 =====================
    df = pd.DataFrame(records)

    # 按 alpha 和样本量汇总
    summary = df.groupby(['alpha', 'sample']).agg(
        avg_fp_count=('fp_count', 'mean'),
        avg_pmono_mean=('fp_pmono_mean', 'mean'),
        avg_pmono_gap=('fp_pmono_gap', 'mean'),
        avg_bdeu_rank=('fp_bdeu_rank_mean', 'mean'),
        avg_bdeu_rank_median=('fp_bdeu_rank_median', 'mean'),
        total_cases=('fp_count', 'count')
    ).reset_index()

    print("===== 被误伤真实边特征汇总 =====")
    print(summary.to_string(index=False))

    # 保存详细数据
    df.to_csv(os.path.join(OUTPUT_DIR, "false_positive_true_edges.csv"), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, "false_positive_true_edges_summary.csv"), index=False)
    print(f"\n详细数据已保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()