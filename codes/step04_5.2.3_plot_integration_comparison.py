#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_integration_comparison.py
生成 5.2.3 集成方案对比图：三种方案（after/before/score）在 ESS=10, α=0.9 下的 TP/FP 折线图。
输出 PDF 图片和对应的 CSV 数据文件。
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

# ===================== 可调参数 =====================
ALPHA = 0.9
BETA = 100
ESS = 10.0
SAMPLE_SIZES = [200, 500, 1000, 2000, 5000, 10000]

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

# ===================== 提取三种方案的数据 =====================
SCHEMES = [
    ('hierarchy_monotonic_after', 'After (Post-hoc)', 'blue', 's', '-'),
    ('hierarchy_monotonic_before', 'Before (Blacklist)', 'red', 'o', '-'),
    ('hierarchy_monotonic_score', 'Score (Soft Penalty)', 'green', '^', '--'),
]

def get_series(base, ess, alpha, beta):
    sub = df[df['base'] == base]
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

# ===================== 存储CSV数据 =====================
csv_rows = []

# ===================== 绘图 =====================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ---- TP 子图 ----
ax = axes[0]
for base, label, color, marker, ls in SCHEMES:
    tp, fp = get_series(base, ESS, ALPHA, BETA if 'score' in base else None)
    if tp is None:
        continue
    ax.plot(SAMPLE_SIZES, tp, color=color, marker=marker, linestyle=ls,
            linewidth=1.5, markersize=8, label=label)
    for i, m in enumerate(SAMPLE_SIZES):
        csv_rows.append({
            'Scheme': label,
            'Sample_Size': m,
            'TP': tp.iloc[i] if not pd.isna(tp.iloc[i]) else np.nan,
            'FP': fp.iloc[i] if not pd.isna(fp.iloc[i]) else np.nan
        })

ax.set_xscale('log')
ax.set_xlabel('Sample Size (m)', fontsize=12)
ax.set_ylabel('True Positives (TP)', fontsize=12)
ax.set_title('TP Comparison (ESS=10, α=0.9)', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_xticks(SAMPLE_SIZES)
ax.set_xticklabels([str(s) for s in SAMPLE_SIZES])
ax.legend(fontsize=10)

# ---- FP 子图 ----
ax = axes[1]
for base, label, color, marker, ls in SCHEMES:
    tp, fp = get_series(base, ESS, ALPHA, BETA if 'score' in base else None)
    if fp is None:
        continue
    ax.plot(SAMPLE_SIZES, fp, color=color, marker=marker, linestyle=ls,
            linewidth=1.5, markersize=8, label=label)

ax.set_xscale('log')
ax.set_xlabel('Sample Size (m)', fontsize=12)
ax.set_ylabel('False Positives (FP)', fontsize=12)
ax.set_title('FP Comparison (ESS=10, α=0.9)', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_xticks(SAMPLE_SIZES)
ax.set_xticklabels([str(s) for s in SAMPLE_SIZES])
ax.legend(fontsize=10)

plt.suptitle('Integration Scheme Comparison', fontsize=14)
plt.tight_layout()

# ===================== 保存PDF图片 =====================
alpha_tag = str(ALPHA).replace('.', 'd')
pdf_path = os.path.join(OUTPUT_DIR, f"integration_comparison_ess{int(ESS)}_alpha{alpha_tag}.pdf")
plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"图片已保存到 {pdf_path}")

# ===================== 保存CSV数据 =====================
csv_df = pd.DataFrame(csv_rows)
csv_df = csv_df.sort_values(['Scheme', 'Sample_Size'])
csv_path = os.path.join(OUTPUT_DIR, f"integration_comparison_ess{int(ESS)}_alpha{alpha_tag}.csv")
csv_df.to_csv(csv_path, index=False)
print(f"数据已保存到 {csv_path}")