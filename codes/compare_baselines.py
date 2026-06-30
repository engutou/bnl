#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_baselines_tp_fp_full.py
绘制基线方法的 TP（真阳性）和 FP（假阳性）绝对值折线图，覆盖有向和骨架两个层面。
布局：2×2 子图
  左上：有向 TP (Directed True Positives)
  右上：有向 FP (Directed False Positives)
  左下：骨架 TP (Skeleton True Positives)
  右下：骨架 FP (Skeleton False Positives)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== 路径配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
CSV_PATH = os.path.join(EXPERIMENTS_DIR, "evaluation_summary-v3.csv")
OUTPUT_DIR = os.path.join(EXPERIMENTS_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 数据加载与方法识别 =====================
df = pd.read_csv(CSV_PATH)


def identify_method(s):
    if s == 'PC':
        return 'PC'
    if s == 'GES_ess1':
        return 'GES_ESS1'
    if s == 'GES_ess10':
        return 'GES_ESS10'
    if s == 'GES':
        return 'GES_ESS1'
    if s == 'HC_BDeu_ess1':
        return 'HC_BDeu_ESS1'
    if s == 'HC_BDeu_ess10':
        return 'HC_BDeu_ESS10'
    return None


df['method'] = df['strategy'].apply(identify_method)
df = df[df['method'].notna()].copy()

# 需要的列：有向和骨架的 TP、FP
cols_needed = ['method', 'sample_size', 'TP', 'FP', 'skeleton_TP', 'skeleton_FP']
df = df[cols_needed]

# ===================== 聚合计算 =====================
grouped = df.groupby(['method', 'sample_size']).mean().reset_index()

# ===================== 绘图参数 =====================
METHODS = ['PC', 'HC_BDeu_ESS1', 'HC_BDeu_ESS10', 'GES_ESS1', 'GES_ESS10']
COLORS = {
    'PC': 'gray',
    'HC_BDeu_ESS1': 'blue',
    'HC_BDeu_ESS10': 'cyan',
    'GES_ESS1': 'green',
    'GES_ESS10': 'lime'
}
MARKERS = {
    'PC': 's',
    'HC_BDeu_ESS1': 'o',
    'HC_BDeu_ESS10': '^',
    'GES_ESS1': 'D',
    'GES_ESS10': 'v'
}
LINE_STYLES = {
    'PC': '--',
    'HC_BDeu_ESS1': '-',
    'HC_BDeu_ESS10': '-',
    'GES_ESS1': '-',
    'GES_ESS10': '-'
}
SAMPLE_SIZES = [100, 200, 500, 1000, 5000, 10000]
GROUND_TRUTH_EDGES = 35

# ===================== 创建 2×2 子图 =====================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 定义四个子图的配置：(子图对象, 数据列, 标题, y轴标签, 理想线)
subplot_configs = [
    (axes[0, 0], 'TP', 'Directed TP (True Positives)',
     'Number of True Positive Edges', GROUND_TRUTH_EDGES),
    (axes[0, 1], 'FP', 'Directed FP (False Positives)',
     'Number of False Positive Edges', 0),
    (axes[1, 0], 'skeleton_TP', 'Skeleton TP (True Positives)',
     'Number of True Positive Edges', GROUND_TRUTH_EDGES),
    (axes[1, 1], 'skeleton_FP', 'Skeleton FP (False Positives)',
     'Number of False Positive Edges', 0),
]

for ax, col, title, ylabel, ideal_line in subplot_configs:
    for method in METHODS:
        sub = grouped[grouped['method'] == method]
        if sub.empty:
            continue
        sub_sorted = sub.set_index('sample_size').reindex(SAMPLE_SIZES).reset_index()
        ax.plot(sub_sorted['sample_size'], sub_sorted[col],
                color=COLORS.get(method, 'black'),
                marker=MARKERS.get(method, 'o'),
                linestyle=LINE_STYLES.get(method, '-'),
                linewidth=2, markersize=8,
                label=method)

    # 标注理想线
    if ideal_line == GROUND_TRUTH_EDGES:
        ax.axhline(y=ideal_line, color='black', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(110, ideal_line + 1.5, f'True edges = {GROUND_TRUTH_EDGES}',
                fontsize=9, color='black', alpha=0.7)
    elif ideal_line == 0:
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    ax.set_xscale('log')
    ax.set_xlabel('Sample Size (m)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(SAMPLE_SIZES)
    ax.set_xticklabels([str(s) for s in SAMPLE_SIZES])

    # 图例位置：TP图放左上，FP图放右上
    if 'TP' in col:
        ax.legend(loc='upper left', fontsize=9)
    else:
        ax.legend(loc='upper right', fontsize=9)

plt.suptitle('Baseline Methods: Absolute TP and FP (Directed & Skeleton) vs Sample Size',
             fontsize=14)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "baselines_tp_fp_full.png")
plt.savefig(save_path, dpi=300)
plt.show()
print(f"图已保存至 {save_path}")

# ===================== 输出表格 =====================
ordered_methods = [m for m in METHODS if m in grouped['method'].unique()]

for level, tp_col, fp_col in [('Directed', 'TP', 'FP'),
                              ('Skeleton', 'skeleton_TP', 'skeleton_FP')]:
    print(f"\n{'=' * 60}")
    print(f"  {level} Level")
    print(f"{'=' * 60}")

    pivot_tp = grouped.pivot(index='method', columns='sample_size', values=tp_col)
    pivot_tp = pivot_tp.reindex(index=ordered_methods, columns=SAMPLE_SIZES)
    print(f"\n--- {level} TP ---")
    print(pivot_tp.to_string(float_format=lambda x: f"{x:.1f}"))

    pivot_fp = grouped.pivot(index='method', columns='sample_size', values=fp_col)
    pivot_fp = pivot_fp.reindex(index=ordered_methods, columns=SAMPLE_SIZES)
    print(f"\n--- {level} FP ---")
    print(pivot_fp.to_string(float_format=lambda x: f"{x:.1f}"))

    # 精确率
    pivot_prec = pivot_tp / (pivot_tp + pivot_fp)
    print(f"\n--- {level} Precision ---")
    print(pivot_prec.to_string(float_format=lambda x: f"{x:.3f}"))

    # 保存CSV
    pivot_tp.to_csv(os.path.join(OUTPUT_DIR, f"baselines_{level.lower()}_tp.csv"))
    pivot_fp.to_csv(os.path.join(OUTPUT_DIR, f"baselines_{level.lower()}_fp.csv"))
    pivot_prec.to_csv(os.path.join(OUTPUT_DIR, f"baselines_{level.lower()}_precision.csv"))

print(f"\n所有图表和数据已保存至 {OUTPUT_DIR}")