#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_beta_sweet_spot_v2.py
横轴：样本量 (Sample Size)
纵轴：TP 或 FP
多条折线：不同 beta 值（beta=0 为 HC_BDeu，beta=inf 为 monotonic_before）
布局：2×2 子图（ESS=1/10 × TP/FP）
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ===================== 路径配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
CSV_PATH = os.path.join(EXPERIMENTS_DIR, "evaluation_summary-v3.csv")
OUTPUT_DIR = os.path.join(EXPERIMENTS_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 固定参数 =====================
ALPHA = 0.3
BETA_VALS = [0, 10]  # 0 代表 HC_BDeu，inf 代表 mono_before (画在最右侧外)
SAMPLE_SIZES = [100, 200, 500, 1000, 5000, 10000]

# ===================== 数据加载与解析 =====================
df = pd.read_csv(CSV_PATH)


def parse_strategy(s):
    base = re.sub(r'_(ess|alpha|beta).*$', '', s)
    ess = re.search(r'_ess(\d+)', s)
    alpha = re.search(r'_alpha([\dd]+)', s)
    beta = re.search(r'_beta(\d+)', s)
    return (base,
            float(ess.group(1)) if ess else None,
            float(alpha.group(1).replace('d', '.')) if alpha else None,
            float(beta.group(1)) if beta else None)


df[['base', 'ess', 'alpha', 'beta']] = df['strategy'].apply(lambda s: pd.Series(parse_strategy(s)))


# ===================== 辅助函数：获取指定策略的 TP、FP =====================
def get_tp_fp(base_name, ess_val, alpha_val=None, beta_val=None):
    """返回 DataFrame，列为 sample_size, TP, FP"""
    sub = df[df['base'] == base_name]
    sub = sub[sub['ess'] == ess_val]
    if alpha_val is not None:
        sub = sub[sub['alpha'] == alpha_val]
    if beta_val is not None:
        sub = sub[sub['beta'] == beta_val]
    if sub.empty:
        return None
    agg = sub.groupby('sample_size')[['TP', 'FP']].mean().reset_index()
    return agg


# ===================== 绘图 =====================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for col, ess_val in enumerate([1.0, 10.0]):
    for row, metric in enumerate(['TP', 'FP']):
        ax = axes[row, col]

        # 绘制 beta=0 到 beta=100 的曲线（monotonic_score）
        for beta in BETA_VALS:
            if beta == 0:
                # 使用 HC_BDeu 作为 beta=0
                curve = get_tp_fp('HC_BDeu', ess_val)
                label = 'β=0 (HC_BDeu)'
                color = 'black'
                linestyle = ':'
                marker = 'x'
            else:
                curve = get_tp_fp('monotonic_score', ess_val, alpha_val=ALPHA, beta_val=beta)
                label = f'β={beta}'
                # 颜色从浅到深
                color_idx = BETA_VALS.index(beta) / len(BETA_VALS)
                color = plt.cm.Blues(0.3 + 0.6 * color_idx)
                linestyle = '-'
                marker = 'o'

            if curve is not None:
                curve_sorted = curve.set_index('sample_size').reindex(SAMPLE_SIZES)
                ax.plot(SAMPLE_SIZES, curve_sorted[metric],
                        color=color, linestyle=linestyle, marker=marker,
                        linewidth=2, markersize=7, label=label)

        # 绘制硬禁止参考线（beta=inf）
        hard_ref = get_tp_fp('monotonic_before', ess_val, alpha_val=ALPHA)
        if hard_ref is not None:
            hard_sorted = hard_ref.set_index('sample_size').reindex(SAMPLE_SIZES)
            ax.plot(SAMPLE_SIZES, hard_sorted[metric],
                    color='red', linestyle='--', marker='D',
                    linewidth=2, markersize=8, label='β=∞ (Hard)')

        ax.set_xscale('log')
        ax.set_xlabel('Sample Size (m)', fontsize=11)
        ax.set_ylabel(f'{metric} (Number of Edges)', fontsize=11)
        ax.set_title(f'ESS={int(ess_val)}, {metric}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(SAMPLE_SIZES)
        ax.set_xticklabels([str(s) for s in SAMPLE_SIZES])
        ax.legend(fontsize=7, loc='best')

plt.suptitle(f'Soft Penalty Sweet Spot Analysis (α={ALPHA})', fontsize=14)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, f"beta_sweet_spot_alpha{str(ALPHA).replace('.', 'd')}.png")
plt.savefig(save_path, dpi=300)
plt.show()
print(f"图已保存至 {save_path}")