#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_ablation_delta.py
绘制五组消融对比的 TP/FP 差值图。
每组对比一张图，每张图包含4个子图（定向/骨架 × TP/FP）。
每条曲线对应一个ESS（1或10）。
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
ALPHA = 0.7  # 单调性阈值
BETA = 10  # 软惩罚默认beta
SAMPLE_SIZES = [100, 200, 500, 1000, 5000, 10000]

# ===================== 数据加载与解析 =====================
df = pd.read_csv(CSV_PATH)


def parse_strategy(s):
    """解析策略字符串，提取base、ess、alpha、beta"""
    base = re.sub(r'_(ess|alpha|beta).*$', '', s)
    ess = re.search(r'_ess(\d+)', s)
    alpha = re.search(r'_alpha([\dd]+)', s)
    beta = re.search(r'_beta(\d+)', s)
    return (base,
            float(ess.group(1)) if ess else None,
            float(alpha.group(1).replace('d', '.')) if alpha else None,
            float(beta.group(1)) if beta else None)


df[['base', 'ess', 'alpha', 'beta']] = df['strategy'].apply(lambda s: pd.Series(parse_strategy(s)))


# ===================== 辅助函数：获取指定策略的指标均值 =====================
def get_metrics(base_name, ess_val, alpha_val=None, beta_val=None):
    """返回 {sample_size: {'TP': ..., 'FP': ..., 'skeleton_TP': ..., 'skeleton_FP': ...}}"""
    sub = df[df['base'] == base_name]
    sub = sub[sub['ess'] == ess_val]
    if alpha_val is not None:
        sub = sub[sub['alpha'] == alpha_val]
    if beta_val is not None:
        sub = sub[sub['beta'] == beta_val]
    if sub.empty:
        return None
    agg = sub.groupby('sample_size')[['TP', 'FP', 'skeleton_TP', 'skeleton_FP']].mean()
    return agg.to_dict(orient='index')


# ===================== 五组对比定义 =====================
COMPARISONS = [
    {
        'title': 'Contrast 0: HC_BDeu vs hierarchy\n(Value of hierarchical constraint)',
        'baseline': ('HC_BDeu', None, None),
        'challenger': ('hierarchy', None, None),
        'tag': 'contrast_0'
    },
    {
        'title': 'Contrast 1: HC_BDeu vs monotonic_before\n(Value of monotonicity constraint alone)',
        'baseline': ('HC_BDeu', None, None),
        'challenger': ('monotonic_before', ALPHA, None),
        'tag': 'contrast_1'
    },
    {
        'title': 'Contrast 2: monotonic_before vs monotonic_score\n(Soft penalty vs hard blacklist)',
        'baseline': ('monotonic_before', ALPHA, None),
        'challenger': ('monotonic_score', ALPHA, BETA),
        'tag': 'contrast_2'
    },
    {
        'title': 'Contrast 3: hierarchy vs hierarchy_mono_before\n(Monotonicity adds value beyond hierarchy)',
        'baseline': ('hierarchy', None, None),
        'challenger': ('hierarchy_monotonic_before', ALPHA, None),
        'tag': 'contrast_3'
    },
    {
        'title': 'Contrast 4: hierarchy_mono_before vs hierarchy_mono_score\n(Soft penalty with hierarchy)',
        'baseline': ('hierarchy_monotonic_before', ALPHA, None),
        'challenger': ('hierarchy_monotonic_score', ALPHA, BETA),
        'tag': 'contrast_4'
    }
]

# ===================== 绘图 =====================
METRICS = ['TP', 'FP', 'skeleton_TP', 'skeleton_FP']
METRIC_LABELS = {
    'TP': 'Δ TP (Directed)',
    'FP': 'Δ FP (Directed)',
    'skeleton_TP': 'Δ TP (Skeleton)',
    'skeleton_FP': 'Δ FP (Skeleton)'
}
ESS_STYLES = {
    1.0: {'color': 'blue', 'linestyle': '-', 'marker': 'o', 'label': 'ESS=1'},
    10.0: {'color': 'red', 'linestyle': '--', 'marker': 's', 'label': 'ESS=10'}
}

for comp in COMPARISONS:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    base_name, base_alpha, base_beta = comp['baseline']
    chal_name, chal_alpha, chal_beta = comp['challenger']

    for idx, metric in enumerate(METRICS):
        ax = axes_flat[idx]

        for ess_val in [1.0, 10.0]:
            base_data = get_metrics(base_name, ess_val, base_alpha, base_beta)
            chal_data = get_metrics(chal_name, ess_val, chal_alpha, chal_beta)

            if base_data is None or chal_data is None:
                continue

            deltas = []
            for m in SAMPLE_SIZES:
                if m in base_data and m in chal_data:
                    delta = chal_data[m][metric] - base_data[m][metric]
                else:
                    delta = np.nan
                deltas.append(delta)

            style = ESS_STYLES[ess_val]
            ax.plot(SAMPLE_SIZES, deltas,
                    color=style['color'], linestyle=style['linestyle'],
                    marker=style['marker'], linewidth=2, markersize=8,
                    label=style['label'])

        # 零线
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.set_xscale('log')
        ax.set_xlabel('Sample Size (m)', fontsize=10)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(SAMPLE_SIZES)
        ax.set_xticklabels([str(s) for s in SAMPLE_SIZES])
        ax.legend(fontsize=8)

    fig.suptitle(comp['title'], fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"ablation_{comp['tag']}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存 {save_path}")

print("\n所有消融图已生成。")