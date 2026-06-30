#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_sensitivity_beta.py
绘制 hierarchy_mono_score 在不同 β 下的 TP 和 FP 变化曲线。
固定 α=0.9, ESS=10。展示 m=200, 500, 2000, 5000 四条曲线。
水平虚线标注 hierarchy_mono_before（硬禁止）的参考值。
输出 PDF 图片和 CSV 数据文件。
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== 路径配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments_N24_E35")
CSV_PATH = os.path.join(EXPERIMENTS_DIR, "evaluation_summary-v3.csv")
OUTPUT_DIR = os.path.join(EXPERIMENTS_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 参数设置 =====================
ESS = 10.0
ALPHA = 0.9
SAMPLE_FOCUS = [200, 500, 2000, 5000]
BETA_VALUES = [10, 30, 50, 70, 100, 200, 500, 1000]

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

# ===================== 提取 hierarchy_mono_score 的数据 =====================
sub = df[(df['base'] == 'hierarchy_monotonic_score') &
         (df['ess'] == ESS) &
         (df['alpha'] == ALPHA)]
if sub.empty:
    print("No data found for hierarchy_monotonic_score. Check CSV path and parameters.")
    exit()

# 按 beta 和 sample_size 聚合
agg = sub.groupby(['beta', 'sample_size'])[['TP', 'FP']].mean().reset_index()

# ===================== 提取硬禁止参考线 =====================
hard = df[(df['base'] == 'hierarchy_monotonic_before') &
          (df['ess'] == ESS) &
          (df['alpha'] == ALPHA)]
hard_agg = hard.groupby('sample_size')[['TP', 'FP']].mean().reset_index() if not hard.empty else None

# ===================== 存储 CSV 数据 =====================
csv_rows = []

# ===================== 绘图 =====================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = {200: '#2166AC', 500: '#67A9CF', 2000: '#EF8A62', 5000: '#B2182B'}
markers = {200: 'o', 500: 's', 2000: '^', 5000: 'D'}

for metric, ax, ylabel, title in [
    ('TP', axes[0], 'True Positives (TP)', 'TP vs $\\beta$'),
    ('FP', axes[1], 'False Positives (FP)', 'FP vs $\\beta$')
]:
    for m in SAMPLE_FOCUS:
        curve = agg[agg['sample_size'] == m].sort_values('beta')
        if curve.empty:
            continue
        ax.plot(curve['beta'], curve[metric],
                color=colors[m], marker=markers[m], linestyle='-',
                linewidth=2, markersize=8, label=f'$m={m}$')
        # 存储数据
        for _, row in curve.iterrows():
            csv_rows.append({
                'Metric': metric,
                'Sample_Size': m,
                'Beta': row['beta'],
                'Value': row[metric]
            })
        # # 标注硬禁止参考线
        # if hard_agg is not None:
        #     hard_m = hard_agg[hard_agg['sample_size'] == m]
        #     if not hard_m.empty:
        #         hard_val = hard_m[metric].values[0]
        #         ax.axhline(y=hard_val, color=colors[m], linestyle='--',
        #                    alpha=0.5, linewidth=1)
        #         # 存储硬禁止参考值
        #         csv_rows.append({
        #             'Metric': f'{metric}_hard',
        #             'Sample_Size': m,
        #             'Beta': 'inf',
        #             'Value': hard_val
        #         })

    ax.set_xscale('log')
    ax.set_xlabel('$\\beta$', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks(BETA_VALUES)
    ax.set_xticklabels([str(b) for b in BETA_VALUES])

plt.suptitle('Sensitivity to $\\beta$ (hierarchy\\_mono\\_score, $\\alpha=0.9$, ESS=10)', fontsize=14)
plt.tight_layout()

# ===================== 保存 PDF =====================
pdf_path = os.path.join(OUTPUT_DIR, "sensitivity_beta.pdf")
plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to {pdf_path}")

# ===================== 保存 CSV =====================
csv_df = pd.DataFrame(csv_rows)
csv_save_path = os.path.join(OUTPUT_DIR, "sensitivity_beta.csv")
csv_df.to_csv(csv_save_path, index=False)
print(f"Data saved to {csv_save_path}")