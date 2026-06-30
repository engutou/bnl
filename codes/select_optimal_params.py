#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_optimal_params.py
从 evaluation_summary-v3.csv 中找出 hierarchy_mono_score 在不同 α、β 下的平均 SHD，
判断现有超参数范围是否足够，并推荐默认值。
"""

import pandas as pd
import numpy as np
import re

# ===================== 配置 =====================
CSV_PATH = "../experiments/evaluation_summary-v3.csv"

# ===================== 数据加载与解析 =====================
df = pd.read_csv(CSV_PATH)

# 解析策略名
def parse_strategy(s):
    base = re.sub(r'_(ess|alpha|beta).*$', '', s)
    ess_match = re.search(r'_ess(\d+)', s)
    alpha_match = re.search(r'_alpha([\dd]+)', s)
    beta_match = re.search(r'_beta(\d+)', s)
    ess = float(ess_match.group(1)) if ess_match else None
    alpha = float(alpha_match.group(1).replace('d', '.')) if alpha_match else None
    beta = float(beta_match.group(1)) if beta_match else None
    return base, ess, alpha, beta

df[['base_strategy', 'ess', 'alpha', 'beta']] = df['strategy'].apply(
    lambda s: pd.Series(parse_strategy(s))
)

# 只关注 hierarchy_mono_score（即 hierarchy_monotonic_score）
mono = df[df['base_strategy'] == 'hierarchy_monotonic_score'].copy()

# ===================== 按 α, β, ESS, 样本量 聚合 SHD =====================
agg = mono.groupby(['alpha', 'beta', 'ess', 'sample_size'])['SHD'].agg(
    mean_SHD='mean', std_SHD='std', count='count'
).reset_index()

# ===================== 总体最优（跨所有样本量平均） =====================
overall = agg.groupby(['alpha', 'beta', 'ess'])['mean_SHD'].mean().reset_index()
overall = overall.sort_values('mean_SHD')
print("===== 跨样本量平均 SHD 最优的 10 个 (α, β, ESS) 组合 =====")
print(overall.head(10).to_string(index=False))

# ===================== 按 ESS 分别看 =====================
for ess_val in [1.0, 10.0]:
    sub = overall[overall['ess'] == ess_val]
    if sub.empty:
        continue
    print(f"\n===== ESS={int(ess_val)} 最优 5 组合 =====")
    print(sub.head(5).to_string(index=False))

# ===================== 检测是否在边界 =====================
alpha_range = agg['alpha'].unique()
beta_range = agg['beta'].unique()
print(f"\n现有 α 范围: {sorted(alpha_range)}")
print(f"现有 β 范围: {sorted(beta_range)}")

best_overall = overall.iloc[0]
print(f"\n最优组合: α={best_overall['alpha']}, β={best_overall['beta']}, ESS={best_overall['ess']}")

need_extend_alpha = best_overall['alpha'] == max(alpha_range)
need_extend_beta = best_overall['beta'] == min(beta_range) or best_overall['beta'] == max(beta_range)

if need_extend_alpha:
    print("⚠ 最优 α 在现有范围上界，建议扩大 α 范围 (如 0.4, 0.5, 0.6, 0.7)。")
else:
    print("✓ 最优 α 在现有范围内。")

if need_extend_beta:
    print("⚠ 最优 β 在现有范围边界，建议扩大 β 范围 (如 10, 30, 50, 70, 1500)。")
else:
    print("✓ 最优 β 在现有范围内。")

# ===================== 按小样本 (m<=500) 和大样本 (m>=5000) 分别看 =====================
small_samples = [100, 200, 500]
large_samples = [5000, 10000]

for name, sample_list in [("小样本 (m≤500)", small_samples), ("大样本 (m≥5000)", large_samples)]:
    sub = agg[agg['sample_size'].isin(sample_list)]
    overall_sub = sub.groupby(['alpha', 'beta', 'ess'])['mean_SHD'].mean().reset_index().sort_values('mean_SHD')
    print(f"\n===== {name} 最优 5 组合 =====")
    print(overall_sub.head(5).to_string(index=False))