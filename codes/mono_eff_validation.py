#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_mono_vs_hier.py
加载 evaluation_summary-v3.csv，对比 hierarchy 和 hierarchy_mono_before
在不同 ESS 和 alpha 下的性能（SHD、F1 等）。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(csv_path):
    """加载评估汇总 CSV，并解析策略中的超参数。"""
    df = pd.read_csv(csv_path)
    # 提取 ess, alpha, beta 信息（如果缺失则填充默认值）
    # 例如策略名 "hierarchy_ess10" -> base_strategy='hierarchy', ess=10
    # "hierarchy_monotonic_before_ess1_alpha0d1" -> base_strategy='hierarchy_monotonic_before', ess=1, alpha=0.1
    # 注意 alpha 格式是 0d1 表示 0.1
    df['base_strategy'] = df['strategy'].str.replace(r'_(ess|alpha|beta).*$', '', regex=True)
    df['ess'] = df['strategy'].str.extract(r'_ess(\d+)').astype(float)
    df['alpha'] = df['strategy'].str.extract(r'_alpha([\dd]+)')
    # 转换 alpha: 将 '0d1' 转换为 0.1
    df['alpha'] = df['alpha'].str.replace('d', '.', regex=False).astype(float)
    # beta 可选
    df['beta'] = df['strategy'].str.extract(r'_beta(\d+)').astype(float)
    return df

def compare_hier_vs_mono(df, metric='SHD', sample_sizes=None):
    """
    对比 hierarchy 和 hierarchy_monotonic_before 在不同 ess 和 alpha 下的指标。
    返回聚合后的表格。
    """
    # 过滤出需要的策略
    target = df[df['base_strategy'].isin(['hierarchy', 'hierarchy_monotonic_before'])]
    if sample_sizes is not None:
        target = target[target['sample_size'].isin(sample_sizes)]

    # 分组：bn_instance, sample_size, ess, alpha (如果有)
    # 注意 hierarchy 没有 alpha 参数，但我们将其 alpha 视为 NaN，聚合时仍能对比
    group_cols = ['bn_instance', 'sample_size', 'ess']
    # 对于 hierarchy_monotonic_before 还有 alpha，但 hierarchy 没有，所以按 alpha 分别聚合后再对比可能复杂。
    # 我们可以先将 hierarchy 的行复制多份，赋予不同的 alpha 以便一一对比？或者更简单：
    # 因为 hierarchy 不依赖 alpha，我们可以对每个 alpha 值，分别与 hierarchy 对比，展示平均差值。
    # 这里我们选择展示 hierarchy 的整体性能（不区分 alpha），而 hierarchy_mono_before 按 alpha 分组。
    # 但为了严格对比，应保持在同一 bn_instance, sample_size, ess 下。所以我们将 hierarchy 的数据与 hierarchy_mono_before 按相同 ess、sample_size 合并。
    # 方法：先分别聚合 hierarchy 和 hierarchy_mono_before 的均值，然后合并比较。
    hier = target[target['base_strategy'] == 'hierarchy']
    mono = target[target['base_strategy'] == 'hierarchy_monotonic_before']

    # 聚合 hierarchy（对每个 bn_instance, sample_size, ess 求平均，但通常每个组合只有一行，直接使用即可）
    hier_avg = hier.groupby(['bn_instance', 'sample_size', 'ess'])[metric].mean().reset_index()
    hier_avg.rename(columns={metric: 'hier_' + metric}, inplace=True)

    # 聚合 mono，按 bn_instance, sample_size, ess, alpha 分组
    mono_avg = mono.groupby(['bn_instance', 'sample_size', 'ess', 'alpha'])[metric].mean().reset_index()
    mono_avg.rename(columns={metric: 'mono_' + metric}, inplace=True)

    # 合并：左连接，因为 hierarchy 没有 alpha，合并后会为每个 alpha 重复 hierarchy 的值
    merged = pd.merge(mono_avg, hier_avg, on=['bn_instance', 'sample_size', 'ess'], how='left')
    merged['diff'] = merged['hier_' + metric] - merged['mono_' + metric]  # 正值表示 mono 改善

    return merged

def print_summary(merged, metric='SHD'):
    """打印按 ess 和 alpha 汇总的平均改善。"""
    summary = merged.groupby(['ess', 'alpha']).agg(
        mean_diff=('diff', 'mean'),
        std_diff=('diff', 'std'),
        mean_hier=('hier_' + metric, 'mean'),
        mean_mono=('mono_' + metric, 'mean'),
        count=('diff', 'count')
    ).reset_index()
    print(f"\n===== 按 ESS 和 Alpha 汇总的平均 {metric} 改善 (hierarchy - mono) =====")
    print(summary.to_string(index=False))
    return summary

def plot_diff(merged, metric='SHD', save_path=None):
    """绘制不同 ess 下，diff 随 alpha 变化的箱线图或曲线。"""
    # 先按样本量分面？为了简洁，我们画两个子图：ess=1 和 ess=10
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, ess_val in zip(axes, [1, 10]):
        sub = merged[merged['ess'] == ess_val]
        if sub.empty:
            ax.set_title(f"ESS={int(ess_val)} (无数据)")
            continue
        # 按 alpha 分组，画 diff 分布
        # 可以画箱线图或均值条形图
        # 由于 alpha 是离散值，这里画条形图表示均值±标准差
        alpha_order = sorted(sub['alpha'].unique())
        means = []
        stds = []
        for a in alpha_order:
            d = sub[sub['alpha'] == a]['diff']
            means.append(d.mean())
            stds.append(d.std())
        x = np.arange(len(alpha_order))
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:.2f}" for a in alpha_order])
        ax.set_xlabel('Alpha')
        ax.set_ylabel(f'{metric} Reduction (hier - mono)')
        ax.set_title(f'ESS = {int(ess_val)}')
        ax.axhline(y=0, color='black', linestyle='--')
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f'Improvement of hierarchy_mono_before over hierarchy ({metric})')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至 {save_path}")
    plt.show()

if __name__ == "__main__":
    CSV_PATH = "../experiments/evaluation_summary-v3.csv"  # 请根据实际路径修改
    df = load_data(CSV_PATH)

    # 1. 验证行数合理性 (期望4530，实际加载行数)
    print(f"加载数据行数: {len(df)}")
    expected = 4530
    if len(df) == expected:
        print("行数符合预期。")
    else:
        print(f"行数与预期({expected})不符，请检查。")

    # 2. 对比 hierarchy_mono_before 和 hierarchy
    # 可指定样本量范围，例如只关注小样本
    sample_sizes = [100, 200, 500, 1000, 5000, 10000]  # 全部
    merged = compare_hier_vs_mono(df, metric='SHD', sample_sizes=sample_sizes)
    print(f"合并后的数据行数: {len(merged)}")

    # 打印摘要表
    summary = print_summary(merged, metric='SHD')

    # 也可对 F1 做同样分析
    # merged_f1 = compare_hier_vs_mono(df, metric='directed_F1', sample_sizes=sample_sizes)
    # print_summary(merged_f1, metric='directed_F1')

    # 画图
    plot_diff(merged, metric='SHD', save_path='mono_vs_hier_diff.png')