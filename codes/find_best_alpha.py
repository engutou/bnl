#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_best_alpha_global.py
在所有实验条件下综合评估，找到全局最优的 alpha。
"""

import os
import re
import pandas as pd
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
CSV_PATH = os.path.join(EXPERIMENTS_DIR, "evaluation_summary-v3.csv")

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

# ===================== 分析策略 =====================
for strategy_name in ['hierarchy_monotonic_before', 'monotonic_before']:
    print(f"\n{'=' * 70}")
    print(f"  Global Best Alpha for: {strategy_name}")
    print(f"{'=' * 70}")

    sub = df[df['base'] == strategy_name]

    # 按 (alpha, sample_size, ess, bn_instance) 聚合
    agg = sub.groupby(['alpha', 'sample_size', 'ess', 'bn_instance'])[['TP', 'FP', 'SHD']].mean().reset_index()

    # 对每个 (sample_size, ess, bn) 组合，计算各α的排名
    # 排名规则：TP越高越好，FP越低越好。综合排名 = TP排名 + FP排名（取平均）
    rankings = []
    for (m, ess_val, bn), grp in agg.groupby(['sample_size', 'ess', 'bn_instance']):
        grp = grp.copy()
        grp['rank_TP'] = grp['TP'].rank(ascending=False)  # TP越高排名越小
        grp['rank_FP'] = grp['FP'].rank(ascending=True)  # FP越低排名越小
        grp['rank_combined'] = (grp['rank_TP'] + grp['rank_FP']) / 2
        rankings.append(
            grp[['alpha', 'sample_size', 'ess', 'bn_instance', 'rank_combined', 'rank_TP', 'rank_FP', 'TP', 'FP']])

    rank_df = pd.concat(rankings)

    # 按α汇总平均排名
    alpha_summary = rank_df.groupby('alpha').agg(
        avg_rank=('rank_combined', 'mean'),
        std_rank=('rank_combined', 'std'),
        avg_TP=('TP', 'mean'),
        avg_FP=('FP', 'mean'),
        avg_rank_TP=('rank_TP', 'mean'),
        avg_rank_FP=('rank_FP', 'mean'),
        n_cases=('rank_combined', 'count')
    ).reset_index()

    # 按平均排名升序排列（越小越好）
    alpha_summary_sorted = alpha_summary.sort_values('avg_rank')

    print("\n--- 各α的综合排名 (越小越好) ---")
    print(alpha_summary_sorted[['alpha', 'avg_rank', 'std_rank', 'avg_TP', 'avg_FP', 'n_cases']].to_string(index=False))

    # 找出最优α
    best_alpha = alpha_summary_sorted.iloc[0]
    print(f"\n>>> 全局最优 α = {best_alpha['alpha']:.2f}")
    print(f"    平均综合排名: {best_alpha['avg_rank']:.2f} ± {best_alpha['std_rank']:.2f}")
    print(f"    平均 TP: {best_alpha['avg_TP']:.1f}, 平均 FP: {best_alpha['avg_FP']:.1f}")

    # 检查是否有统计显著优于第二名
    if len(alpha_summary_sorted) >= 2:
        second_best = alpha_summary_sorted.iloc[1]
        # 对最优α和次优α在所有条件下的rank_combined做配对检验
        best_ranks = rank_df[rank_df['alpha'] == best_alpha['alpha']]['rank_combined'].values
        second_ranks = rank_df[rank_df['alpha'] == second_best['alpha']]['rank_combined'].values
        if len(best_ranks) == len(second_ranks):
            t_stat, p_val = stats.ttest_rel(best_ranks, second_ranks)
            if p_val < 0.05:
                print(f"    (显著优于第二名 α={second_best['alpha']:.2f}, p={p_val:.4f})")
            else:
                print(f"    (与第二名 α={second_best['alpha']:.2f} 无显著差异, p={p_val:.4f})")
                print(f"    → 两者均可作为默认推荐")

    # 帕累托分析：哪些α在avg_TP和avg_FP上都是最优的
    print("\n--- 帕累托前沿 (在 avg_TP 和 avg_FP 上不被其他α支配) ---")
    pareto = []
    for i, row_i in alpha_summary.iterrows():
        dominated = False
        for j, row_j in alpha_summary.iterrows():
            if i == j:
                continue
            # row_j 在 TP 和 FP 上都优于 row_i
            if row_j['avg_TP'] >= row_i['avg_TP'] and row_j['avg_FP'] <= row_i['avg_FP']:
                # 至少有一个严格优于
                if row_j['avg_TP'] > row_i['avg_TP'] or row_j['avg_FP'] < row_i['avg_FP']:
                    dominated = True
                    break
        if not dominated:
            pareto.append(row_i['alpha'])

    pareto_df = alpha_summary[alpha_summary['alpha'].isin(pareto)].sort_values('avg_TP', ascending=False)
    print(pareto_df[['alpha', 'avg_TP', 'avg_FP', 'avg_rank']].to_string(index=False))
    print(f"帕累托前沿共 {len(pareto)} 个α值")