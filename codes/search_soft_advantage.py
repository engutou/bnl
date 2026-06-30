#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_soft_vs_hard_significance.py
按参数组合分组，检验软惩罚相对于硬惩罚的 TP、FP 差异是否统计显著。
"""

import os
import re
import pandas as pd
import numpy as np
from scipy import stats

# ===================== 路径配置 =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
CSV_PATH = os.path.join(EXPERIMENTS_DIR, "evaluation_summary-v3.csv")
OUTPUT_DIR = os.path.join(EXPERIMENTS_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# 筛选两种策略，只保留需要的列
cols = ['bn_instance', 'sample_size', 'ess', 'alpha', 'beta', 'TP', 'FP']
hard = df[df['base'] == 'hierarchy_monotonic_before'][cols].copy()
soft = df[df['base'] == 'hierarchy_monotonic_score'][cols].copy()

# 删除 hard 中的 beta 列（全部为 NaN），避免合并时列名冲突
hard = hard.drop(columns=['beta'])

# 合并：soft 保留 beta，hard 无 beta，合并后 beta 列来自 soft
merged = pd.merge(soft, hard,
                  on=['bn_instance', 'sample_size', 'ess', 'alpha'],
                  suffixes=('_soft', '_hard'))

# 计算差值
merged['delta_TP'] = merged['TP_soft'] - merged['TP_hard']
merged['delta_FP'] = merged['FP_soft'] - merged['FP_hard']

# ===================== 分组统计检验 =====================
grouped = merged.groupby(['sample_size', 'ess', 'alpha', 'beta'])

results = []
for name, group in grouped:
    sample_size, ess_val, alpha_val, beta_val = name
    n_bn = group['bn_instance'].nunique()
    if n_bn < 3:
        continue

    delta_tp = group['delta_TP'].dropna()
    delta_fp = group['delta_FP'].dropna()

    if len(delta_tp) == 0 or len(delta_fp) == 0:
        continue

    t_tp, p_tp = stats.ttest_1samp(delta_tp, popmean=0)
    t_fp, p_fp = stats.ttest_1samp(delta_fp, popmean=0)

    results.append({
        'sample_size': sample_size,
        'ess': ess_val,
        'alpha': alpha_val,
        'beta': beta_val,
        'n_bn': n_bn,
        'mean_delta_TP': delta_tp.mean(),
        'std_delta_TP': delta_tp.std(),
        't_TP': t_tp,
        'p_TP': p_tp,
        'mean_delta_FP': delta_fp.mean(),
        'std_delta_FP': delta_fp.std(),
        't_FP': t_fp,
        'p_FP': p_fp
    })

res_df = pd.DataFrame(results)

# ===================== 输出显著结果 =====================
tp_sig = res_df[(res_df['mean_delta_TP'] > 0) & (res_df['p_TP'] < 0.05)].sort_values('p_TP')
print("===== ΔTP 显著为正 (软惩罚TP更高) =====")
print(tp_sig[['sample_size', 'ess', 'alpha', 'beta', 'n_bn', 'mean_delta_TP', 'p_TP']].to_string(index=False))

fp_sig = res_df[(res_df['mean_delta_FP'] < 0) & (res_df['p_FP'] < 0.05)].sort_values('p_FP')
print("\n===== ΔFP 显著为负 (软惩罚FP更低) =====")
print(fp_sig[['sample_size', 'ess', 'alpha', 'beta', 'n_bn', 'mean_delta_FP', 'p_FP']].to_string(index=False))

combined_sig = res_df[
    (res_df['mean_delta_TP'] > 0) & (res_df['mean_delta_FP'] <= 0) &
    ((res_df['p_TP'] < 0.05) | (res_df['p_FP'] < 0.05))
    ].sort_values('p_TP')
print("\n===== 综合优势 (TP↑, FP↓/→) 且显著 =====")
print(combined_sig[['sample_size', 'ess', 'alpha', 'beta', 'n_bn',
                    'mean_delta_TP', 'p_TP', 'mean_delta_FP', 'p_FP']].to_string(index=False))

res_df.to_csv(os.path.join(OUTPUT_DIR, "soft_vs_hard_significance.csv"), index=False)
print("\n完整统计结果已保存。")