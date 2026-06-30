#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step04_5.2.4_alpha.1_plot_pmono_by_sample.py (updated)
Plot $P_{\mathrm{mono}}$ distribution histograms for true and spurious edges,
separately for each sample size. Reads monotonicity JSON files directly
from the experiments directory to obtain the most accurate $P_{\mathrm{mono}}$ values.
Outputs:
  - pmono_distribution_by_sample.pdf
  - pmono_distribution_data.csv
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== 路径配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments_N24_E35")
BASE_DIR = os.path.join(EXPERIMENTS_DIR, "generated_bns")
OUTPUT_DIR = os.path.join(EXPERIMENTS_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 样本量列表（与实验一致）
SAMPLE_SIZES = ['S200', 'S500', 'S1000', 'S2000', 'S5000', 'S10000']
# 参考 alpha 线
ALPHA_REFS = [0.1, 0.5, 0.9]

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def collect_pmono_data():
    """遍历所有BN实例和样本量，收集$P_{\mathrm{mono}}$值以及是否真实边的标签"""
    records = []
    for bn_dir in sorted(Path(BASE_DIR).glob("bn_*")):
        if not bn_dir.is_dir():
            continue
        # 加载ground truth
        gt_file = bn_dir / "ground_truth" / "dag.json"
        if not gt_file.is_file():
            gt_file = bn_dir / "ground_truth" / "topo.json"
        if not gt_file.is_file():
            continue
        gt_data = load_json(str(gt_file))
        gt_edges = {(e['from'], e['to']) for e in gt_data['edges']['single_edges']}

        data_dir = bn_dir / "data"
        for sample_dir in sorted(data_dir.glob("S*")):
            sample_name = sample_dir.name
            if sample_name not in SAMPLE_SIZES:
                continue
            # 找单调性JSON文件
            mono_files = list(sample_dir.glob("*_mono_blacklist_alpha*.json"))
            if not mono_files:
                continue
            mono_data = load_json(str(mono_files[0]))
            all_pairs = mono_data.get("all_pairs_pmono", [])
            for pair in all_pairs:
                u, v = pair['from'], pair['to']
                p = pair['P_mono']
                is_true = (u, v) in gt_edges
                records.append({
                    'sample': sample_name,
                    'P_mono': p,
                    'is_true': is_true
                })
    return pd.DataFrame(records)

def main():
    df = collect_pmono_data()
    if df.empty:
        print("No P_mono data found. Please run monotonicity generation first.")
        return

    # 保存CSV数据文件
    csv_save_path = os.path.join(OUTPUT_DIR, "pmono_distribution_data.csv")
    df.to_csv(csv_save_path, index=False)
    print(f"Data saved to {csv_save_path}")

    # 启用LaTeX文本渲染
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })

    # 分组绘制
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    bins = np.arange(0, 1.01, 0.05)
    # 柱子偏移量，使两个直方图并排显示，避免重叠
    bar_width = 0.02
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for idx, sample in enumerate(SAMPLE_SIZES):
        ax = axes_flat[idx]
        sub = df[df['sample'] == sample]
        if sub.empty:
            ax.set_title(f"$m={sample[1:]}$ (no data)")
            continue

        true_p = sub[sub['is_true'] == True]['P_mono']
        false_p = sub[sub['is_true'] == False]['P_mono']

        # 手动计算频率密度，使两个直方图错位显示
        hist_true, _ = np.histogram(true_p, bins=bins, density=True)
        hist_false, _ = np.histogram(false_p, bins=bins, density=True)

        # 绘制并排的柱状图，左为虚假边，右为真实边
        ax.bar(bin_centers - bar_width/2, hist_false, width=bar_width,
               alpha=0.7, color='#E74C3C', edgecolor='white', linewidth=0.3,
               label=f'Spurious edges (n={len(false_p)})')
        ax.bar(bin_centers + bar_width/2, hist_true, width=bar_width,
               alpha=0.7, color='#3498DB', edgecolor='white', linewidth=0.3,
               label=f'True edges (n={len(true_p)})')

        # 标注 alpha 参考线
        for alpha in ALPHA_REFS:
            ax.axvline(x=alpha, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.text(alpha + 0.01, ax.get_ylim()[1] * 0.7, f'$\\alpha$={alpha}',
                    fontsize=7, color='gray')

        ax.set_xlabel('$P_{\\mathrm{mono}}$')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'$m={sample[1:]}$')
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)

    # 如果子图数量多于样本量，隐藏多余的子图
    for idx in range(len(SAMPLE_SIZES), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('$P_{\\mathrm{mono}}$ Distribution for True and Spurious Edges', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "pmono_distribution_by_sample.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    main()