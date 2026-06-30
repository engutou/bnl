#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_main_results_figure.py
生成 5.2.1 主结果图：TP 和 FP 随样本量变化的折线图。同时输出对应数据为 CSV 文件。
使用变量控制 alpha 和 ESS，自动清理 legend 中的 LaTeX 转义符。
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== 路径配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXP_FOLDER_NAME = "experiments_N12_E16"
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, EXP_FOLDER_NAME)
CSV_PATH = os.path.join(EXPERIMENTS_DIR, "evaluation_summary-v3.csv")
OUTPUT_DIR = os.path.join(EXPERIMENTS_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 可调参数 =====================
ALPHA = 0.9                 # 单调性阈值，0.7 或 0.9
SAMPLE_SIZES = [200, 500, 1000, 2000, 5000, 10000]
if EXP_FOLDER_NAME == "experiments_N24_E35":
    GROUND_TRUTH = 35           # 24节点网络真实边数
    num_nodes = 24
elif EXP_FOLDER_NAME == "experiments_N12_E16":
    GROUND_TRUTH = 16           # 16节点网络真实边数
    num_nodes = 12

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

# ===================== 方法定义 =====================
# 格式: (显示名称, base, ess, alpha, beta, 颜色, 标记, 线型)
# 显示名称中的下划线直接写，matplotlib 会正常显示，避免使用 \_

METHODS = [('PC', 'PC', None, None, None, 'gray', 's', '--')]
if EXP_FOLDER_NAME == 'experiments_N24_E35':
    METHODS.extend([('GES_ESS1', 'GES', 1.0, None, None, 'green', '^', '-'),
                    ('HC_BDeu_ESS1', 'HC_BDeu', 1.0, None, None, 'blue', 'o', '-'),
                    ('hierarchy_ESS1', 'hierarchy', 1.0, None, None, 'orange', 'D', '-'),
                    ('hierarchy_mono_before_ESS1', 'hierarchy_monotonic_before', 1.0, ALPHA, None, 'red', '*', '-'),
                    ])
METHODS.extend([('GES_ESS10', 'GES', 10.0, None, None, 'lime', 'v', '--'),
                ('HC_BDeu_ESS10', 'HC_BDeu', 10.0, None, None, 'cyan', 's', '--'),
                ('hierarchy_ESS10', 'hierarchy', 10.0, None, None, 'gold', 'D', '--'),
                ('hierarchy_mono_before_ESS10', 'hierarchy_monotonic_before', 10.0, ALPHA, None, 'magenta', '*', '--')])

# ===================== 提取数据 =====================
def get_series(base, ess, alpha, beta):
    sub = df[df['base'] == base]
    if ess is not None:
        sub = sub[sub['ess'] == ess]
    if alpha is not None:
        sub = sub[sub['alpha'] == alpha]
    if beta is not None:
        sub = sub[sub['beta'] == beta]
    if sub.empty:
        return None, None
    agg = sub.groupby('sample_size')[['TP', 'FP']].mean()
    tp = agg['TP'].reindex(SAMPLE_SIZES)
    fp = agg['FP'].reindex(SAMPLE_SIZES)
    return tp, fp

# ===================== 存储数据用于CSV导出 =====================
csv_data = []  # 列表，每个元素是一个字典，包含方法和各样本量的 TP/FP

# ===================== 绘图 =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- TP 子图 ----
ax = axes[0]
for label, base, ess, alpha, beta, color, marker, ls in METHODS:
    tp, fp = get_series(base, ess, alpha, beta)
    if tp is None:
        continue
    ax.plot(SAMPLE_SIZES, tp, color=color, marker=marker, linestyle=ls,
            linewidth=1.5 if 'ESS=1' in label else 1.0,
            markersize=8, label=label)
    # 收集数据
    for i, m in enumerate(SAMPLE_SIZES):
        csv_data.append({
            'Method': label,
            'Sample_Size': m,
            'TP': tp.iloc[i] if not pd.isna(tp.iloc[i]) else np.nan,
            'FP': fp.iloc[i] if not pd.isna(fp.iloc[i]) else np.nan
        })

ax.axhline(y=GROUND_TRUTH, color='black', linestyle=':', alpha=0.5, linewidth=1)
ax.text(220, GROUND_TRUTH - 0.9, f'Total true edges = {GROUND_TRUTH}',
        fontsize=9, color='black', alpha=0.7)
ax.set_xscale('log')
ax.set_xlabel('Sample Size (m)', fontsize=12)
ax.set_ylabel('True Positives (TP)', fontsize=12)
ax.set_title('Discovery: True Positives (higher is better)', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_xticks(SAMPLE_SIZES)
ax.set_xticklabels([str(s) for s in SAMPLE_SIZES])
if EXP_FOLDER_NAME == "experiments_N24_E35":
    ax.legend(fontsize=8, ncol=2, loc="center left",  # 图例自身哪个点对齐目标坐标
              bbox_to_anchor=(0.045, 0.75)  # 目标坐标 (x,y)
              )
else:
    ax.legend(fontsize=8, loc="center left",  # 图例自身哪个点对齐目标坐标
              bbox_to_anchor=(0.045, 0.75)  # 目标坐标 (x,y)
              )

# ---- FP 子图 ----
ax = axes[1]
for label, base, ess, alpha, beta, color, marker, ls in METHODS:
    tp, fp = get_series(base, ess, alpha, beta)
    if fp is None:
        continue
    ax.plot(SAMPLE_SIZES, fp, color=color, marker=marker, linestyle=ls,
            linewidth=1.5 if 'ESS=1' in label else 1.0,
            markersize=8, label=label)

ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax.set_xscale('log')
ax.set_xlabel('Sample Size (m)', fontsize=12)
ax.set_ylabel('False Positives (FP)', fontsize=12)
ax.set_title('False Alarms: False Positives (lower is better)', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_xticks(SAMPLE_SIZES)
ax.set_xticklabels([str(s) for s in SAMPLE_SIZES])
ax.legend(fontsize=8)

alpha_tag = str(ALPHA).replace('.', 'd')
plt.suptitle(f'Main Results: TP and FP vs Sample Size ({num_nodes}-node Network, α={ALPHA})',
             fontsize=14)
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, f"fig_5.1_tp_fp_trends_alpha{alpha_tag}.pdf")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to {save_path}")

# ===================== 导出CSV数据 =====================
csv_df = pd.DataFrame(csv_data)
# 按方法和样本量排序
csv_df = csv_df.sort_values(['Method', 'Sample_Size'])
csv_save_path = os.path.join(OUTPUT_DIR, f"fig_5.1_data_alpha{alpha_tag}.csv")
csv_df.to_csv(csv_save_path, index=False)
print(f"Data saved to {csv_save_path}")
