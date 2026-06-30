#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_beta_insight_enhanced.py
为论文 β 分析提供增强的实验数据：
  - 单条真实边的典型 BDeu 增益
  - 真实边与噪声边的单调性惩罚统计（平均 φ、被惩罚比例等）
  - 不同 β 下先验/似然比值
"""

import os
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from pgmpy.estimators import BDeu
from pgmpy.base import DAG

# ===================== 配置 =====================
BASE_DIR = "../experiments/generated_bns"
ALPHA = 0.3  # 单调性阈值，与实验一致
BETA_VALUES = [100, 200, 500, 1000]
ESS = 10  # BDeu 等效样本大小，与实验一致


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_gt_edges(gt_path):
    """从 Ground Truth JSON 提取真实有向边集合"""
    data = load_json(gt_path)
    edges = data.get('edges', {}).get('single_edges', [])
    return [(e['from'], e['to']) for e in edges]


def compute_bdeu_gain(data, gt_edges):
    """
    计算空图与真图的 BDeu 得分差，并除以真实边数得到单边平均增益。
    返回 (delta, avg_gain_per_edge)
    """
    # 确保数据列名正确
    data_cols = list(data.columns)

    # 空图得分
    empty_dag = DAG()
    empty_dag.add_nodes_from(data_cols)
    scorer = BDeu(data, equivalent_sample_size=ESS)
    score_empty = sum(scorer.local_score(node, []) for node in data_cols)

    # 真图得分
    true_dag = DAG()
    true_dag.add_nodes_from(data_cols)
    for u, v in gt_edges:
        if u not in data_cols or v not in data_cols:
            print(f"Warning: edge {u}->{v} not in data columns, skipping")
            continue
        true_dag.add_edge(u, v)
    score_true = sum(scorer.local_score(node, list(true_dag.predecessors(node)))
                     for node in data_cols)

    delta = score_true - score_empty
    avg_gain = delta / len(gt_edges) if len(gt_edges) > 0 else 0.0
    return delta, avg_gain


def compute_mono_stats(mono_path, gt_edges, alpha=ALPHA):
    """
    从单调性 JSON 计算真实边与噪声边的惩罚统计。
    返回一个包含各项统计的字典。
    """
    data = load_json(mono_path)
    all_pairs = data.get('all_pairs_pmono', [])

    # 构建真实边集合（用于快速判断）
    true_set = set(gt_edges)

    # 存储所有边的 φ 值
    true_phis = []  # 真实边的 φ
    noise_phis = []  # 噪声边的 φ（排除自环）

    for pair in all_pairs:
        u, v = pair['from'], pair['to']
        p = pair['P_mono']
        phi = max(0.0, alpha - p)
        if (u, v) in true_set:
            true_phis.append(phi)
        elif u != v:  # 排除自环，视为噪声边
            noise_phis.append(phi)

    # 统计真实边
    n_true = len(true_phis)
    true_punished = [phi for phi in true_phis if phi > 0]
    true_punished_count = len(true_punished)
    true_avg_phi = np.mean(true_phis) if n_true > 0 else 0.0
    true_punished_ratio = true_punished_count / n_true if n_true > 0 else 0.0
    true_avg_phi_punished = np.mean(true_punished) if true_punished_count > 0 else 0.0

    # 统计噪声边
    n_noise = len(noise_phis)
    noise_punished = [phi for phi in noise_phis if phi > 0]
    noise_punished_count = len(noise_punished)
    noise_avg_phi = np.mean(noise_phis) if n_noise > 0 else 0.0
    noise_punished_ratio = noise_punished_count / n_noise if n_noise > 0 else 0.0
    noise_avg_phi_punished = np.mean(noise_punished) if noise_punished_count > 0 else 0.0

    # 所有违规边（用于兼容原有 avg_phi）
    all_violations = [phi for phi in (true_phis + noise_phis) if phi > 0]
    overall_avg_phi = np.mean(all_violations) if all_violations else 0.0

    return {
        'n_true_edges': n_true,
        'true_avg_phi_all': true_avg_phi,
        'true_punished_count': true_punished_count,
        'true_punished_ratio': true_punished_ratio,
        'true_avg_phi_punished': true_avg_phi_punished,
        'n_noise_edges': n_noise,
        'noise_avg_phi_all': noise_avg_phi,
        'noise_punished_count': noise_punished_count,
        'noise_punished_ratio': noise_punished_ratio,
        'noise_avg_phi_punished': noise_avg_phi_punished,
        'overall_avg_phi': overall_avg_phi  # 所有违规边的平均 φ（用于原有比较）
    }


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
        gt_edges = extract_gt_edges(str(gt_file))

        # 遍历样本量
        data_dir = bn_dir / "data"
        if not data_dir.is_dir():
            continue
        for sample_dir in sorted(data_dir.glob("S*")):
            if not re.match(r'S\d+$', sample_dir.name):
                continue
            sample_name = sample_dir.name

            # CSV 文件
            csv_files = list(sample_dir.glob("*.csv"))
            if not csv_files:
                continue
            csv_path = str(csv_files[0])

            # 单调性 JSON
            mono_files = list(sample_dir.glob("*_mono_blacklist_alpha*.json"))
            if not mono_files:
                continue
            mono_path = str(mono_files[0])

            # 读取数据
            data = pd.read_csv(csv_path, sep=',')

            # 1. BDeu 增益
            delta, avg_gain = compute_bdeu_gain(data, gt_edges)

            # 2. 单调性惩罚统计（区分真实边与噪声边）
            mono_stats = compute_mono_stats(mono_path, gt_edges, alpha=ALPHA)

            # 3. 合并记录
            record = {
                'bn': bn_name,
                'sample': sample_name,
                'num_edges': len(gt_edges),
                'bdeu_delta': round(delta, 2),
                'avg_gain_per_edge': round(avg_gain, 2),
                **mono_stats
            }

            # 4. 不同 β 的比值（基于 overall_avg_phi 或噪声边的惩罚？建议使用 overall）
            avg_phi = mono_stats['overall_avg_phi']
            for b in BETA_VALUES:
                penalty = b * avg_phi
                ratio = penalty / avg_gain if avg_gain > 0 else 0.0
                record[f'beta_{b}_penalty'] = round(penalty, 2)
                record[f'beta_{b}_ratio'] = round(ratio, 4)

            records.append(record)
            print(f"Processed {bn_name}/{sample_name}: gain={avg_gain:.2f}, "
                  f"true_punish_ratio={mono_stats['true_punished_ratio']:.3f}")

    # 保存结果
    df = pd.DataFrame(records)
    df.to_csv("beta_insight_enhanced.csv", index=False)
    print(f"\nResults saved to beta_insight_enhanced.csv ({len(df)} rows)")

    # 打印汇总统计
    print("\n===== Summary Statistics =====")
    print(f"Average BDeu gain per edge: {df['avg_gain_per_edge'].mean():.2f} "
          f"± {df['avg_gain_per_edge'].std():.2f}")
    print(f"Overall average phi (violations): {df['overall_avg_phi'].mean():.4f}")
    print(f"True edge punished ratio: {df['true_punished_ratio'].mean():.4f}")
    print(f"True edge avg phi (when punished): {df['true_avg_phi_punished'].mean():.4f}")
    print(f"Noise edge punished ratio: {df['noise_punished_ratio'].mean():.4f}")
    print(f"Noise edge avg phi (when punished): {df['noise_avg_phi_punished'].mean():.4f}")
    print()
    for b in BETA_VALUES:
        col = f'beta_{b}_ratio'
        print(f"Beta={b}: average ratio = {df[col].mean():.4f} ± {df[col].std():.4f}")


if __name__ == "__main__":
    main()