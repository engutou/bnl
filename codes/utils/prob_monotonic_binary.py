#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Weak Monotonicity Test for Bayesian Network Structure Learning

This script provides a function to compute the posterior probability
P(p1 > p0 | data) given binary observations of a candidate parent-child pair.
It includes a usage example with synthetic data.
"""
import json
import os
from datetime import datetime
from typing import Dict, Union, Optional

import numpy as np
import pandas as pd


def bayesian_weak_monotonicity_test(
        x: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
        num_samples: int = 100000,
        random_seed: Optional[int] = 42,
        neutral_value: float = 0.5
) -> float:
    """
    贝叶斯弱单调性检验：计算 P(p1 > p0 | data) 的后验概率。

    参数
    ----------
    x : array-like
        父节点 X 的观测序列，要求取值为 0 (正常) 或 1 (故障)。
    y : array-like
        子节点 Y 的观测序列，要求取值为 0 (正常) 或 1 (故障)。
        长度必须与 x 相同。
    num_samples : int, default=100000
        蒙特卡洛采样的样本数量。
    random_seed : int, optional, default=42
        随机数种子，保证结果可复现。
    neutral_value : float, default=0.5
        当某组样本数为 0 时返回的中性概率值。

    返回
    -------
    float
        P(p1 > p0 | data) 的估计值，范围 [0, 1]。
    """
    # 转换为 numpy 数组，确保布尔索引正常工作
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    # 可选：检查输入是否为 0/1 二值，若有必要可在此处自动二值化，但建议调用者预先处理
    if not np.all(np.isin(x, [0, 1])):
        raise ValueError("x must contain only 0 and 1 values.")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only 0 and 1 values.")

    # 计算充分统计量
    mask_x0 = (x == 0)
    mask_x1 = (x == 1)

    n0 = np.sum(mask_x0)
    n1 = np.sum(mask_x1)

    # 边界情况：某组样本数为 0，无法有效估计后验
    if n0 == 0 or n1 == 0:
        return neutral_value

    # X=0 时 Y=1 的计数
    k0 = np.sum(y[mask_x0] == 1)
    # X=1 时 Y=1 的计数
    k1 = np.sum(y[mask_x1] == 1)

    # Beta 后验参数 (平坦先验 Beta(1,1))
    alpha0 = 1 + k0
    beta0 = 1 + n0 - k0
    alpha1 = 1 + k1
    beta1 = 1 + n1 - k1

    # 设置随机数生成器
    rng = np.random.default_rng(random_seed)

    # 蒙特卡洛采样
    p0_samples = rng.beta(alpha0, beta0, size=num_samples)
    p1_samples = rng.beta(alpha1, beta1, size=num_samples)

    # 计算 p1 > p0 的比例
    p_mono = np.mean(p1_samples > p0_samples)

    return float(p_mono)


def batch_monotonicity_test_from_csv(
        csv_path: str,
        num_samples: int = 100000,
        random_seed: Optional[int] = 42,
        neutral_value: float = 0.5,
        verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    从 CSV 文件读取多状态样本数据，计算所有有序节点对的弱单调性后验概率。

    假设：
    - CSV 文件第一行为列名（变量名）。
    - 每个单元格为整数值，0 表示正常状态，非 0 表示故障状态。
    - 样本独立同分布，无缺失值。

    参数
    ----------
    csv_path : str
        CSV 文件的完整路径。
    num_samples : int, default=10000
        每对节点蒙特卡洛采样的样本数。
    random_seed : int, optional, default=42
        随机数种子，保证结果可复现。
    neutral_value : float, default=0.5
        当某组样本数为 0 时返回的中性概率值。
    verbose : bool, default=True
        是否打印进度信息。

    返回
    -------
    Dict[str, pd.DataFrame]
        包含两个键的字典：
        - 'pairs': DataFrame，列为 ['Parent', 'Child', 'P_mono']，每一行是一个有序对。
        - 'matrix': DataFrame，以父节点为行、子节点为列的矩阵形式，对角线为 NaN。
    """
    # 1. 读取数据
    if verbose:
        print(f"Loading data from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    node_names = df_raw.columns.tolist()
    n_nodes = len(node_names)

    if verbose:
        print(f"Found {n_nodes} nodes, {len(df_raw)} samples.")

    # 2. 二值化（0 -> 0, 非0 -> 1）
    df_bin = (df_raw != 0).astype(int)

    # 3. 预先计算每个变量的二值数组，加速后续访问
    bin_arrays = {name: df_bin[name].values for name in node_names}

    # 4. 计算每对节点的 n0, k0, n1, k1（向量化操作）
    #    先为所有节点计算 k1 (Y=1) 的总和等，但需要按 X 分组，因此采用循环。
    if verbose:
        print("Computing sufficient statistics for all ordered pairs...")

    stats = {}  # (parent, child) -> (n0, k0, n1, k1)
    for parent in node_names:
        x = bin_arrays[parent]
        mask0 = (x == 0)
        mask1 = (x == 1)
        n0 = mask0.sum()
        n1 = mask1.sum()
        for child in node_names:
            if parent == child:
                continue
            y = bin_arrays[child]
            k0 = y[mask0].sum()
            k1 = y[mask1].sum()
            stats[(parent, child)] = (n0, k0, n1, k1)

    # 5. 对每对节点调用贝叶斯检验（复用之前定义的函数）
    #    注意：这里假设 bayesian_weak_monotonicity_test 已定义（可从外部导入）
    if verbose:
        print("Running Bayesian monotonicity tests...")

    results = []
    total_pairs = len(stats)
    report_interval = max(1, total_pairs // 10)  # 每完成 10% 打印一次

    for idx, ((parent, child), (n0, k0, n1, k1)) in enumerate(stats.items()):
        if n0 == 0 or n1 == 0:
            p_mono = neutral_value
        else:
            p_mono = bayesian_weak_monotonicity_test(
                bin_arrays[parent], bin_arrays[child],
                num_samples=num_samples,
                random_seed=random_seed,
                neutral_value=neutral_value
            )
        results.append((parent, child, p_mono))

        # 打印进度
        if (idx + 1) % report_interval == 0 or (idx + 1) == total_pairs:
            print(f"Progress: {idx + 1}/{total_pairs} pairs processed ({(idx + 1) / total_pairs:.1%})")

    # 6. 构建输出数据结构
    df_pairs = pd.DataFrame(results, columns=['Parent', 'Child', 'P_mono'])

    # 构建矩阵形式
    matrix = pd.DataFrame(np.nan, index=node_names, columns=node_names)
    for _, row in df_pairs.iterrows():
        matrix.loc[row['Parent'], row['Child']] = row['P_mono']

    if verbose:
        print("Done.")

    return {'pairs': df_pairs, 'matrix': matrix}


def __load_ground_truth_from_json(json_path: str) -> pd.DataFrame:
    """
    从指定的 JSON 文件加载 ground truth 拓扑，返回邻接矩阵 (1 表示有向边)。

    参数
    ----------
    json_path : str
        JSON 文件路径，格式需符合提供的示例（包含 individual_variables 和 edges.single_edges）。

    返回
    -------
    pd.DataFrame
        行列均为节点名的方阵，1 表示存在边，0 表示不存在。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取所有节点名（按 individual_variables 的键，可保持排序一致性）
    nodes = list(data['individual_variables'].keys())

    # 初始化全零矩阵
    gt_matrix = pd.DataFrame(0, index=nodes, columns=nodes)

    # 填充边
    for edge in data['edges']['single_edges']:
        u = edge['from']
        v = edge['to']
        if u in nodes and v in nodes:
            gt_matrix.loc[u, v] = 1
        else:
            print(f"Warning: Edge {u} -> {v} contains unknown node, ignored.")

    return gt_matrix


def __evaluate_monotonicity(
        gt_matrix: pd.DataFrame,
        mono_matrix: pd.DataFrame,
        alpha: float = 0.1,
        verbose: bool = True
) -> Dict:
    """
    对比 ground truth 矩阵与单调性检测矩阵，计算混淆矩阵和性能指标。

    参数
    ----------
    gt_matrix : pd.DataFrame
        0/1 矩阵，1 表示边存在于 ground truth。
    mono_matrix : pd.DataFrame
        P_mono 概率矩阵，行列与 gt_matrix 一致。
    alpha : float, default=0.1
        阈值：P_mono < alpha 视为预测为负（无单调性支持），>= alpha 视为预测为正。
    verbose : bool, default=True
        是否打印详细结果。

    返回
    -------
    Dict
        包含 'comparison' (DataFrame), 'metrics' (dict) 等。
    """
    # 确保行列一致
    common_nodes = gt_matrix.columns.intersection(mono_matrix.columns)
    if len(common_nodes) == 0:
        raise ValueError("No common nodes between ground truth and monotonicity matrix.")

    # 转换为浮点类型，以便容纳 NaN
    gt = gt_matrix.loc[common_nodes, common_nodes].astype(float)
    mono = mono_matrix.loc[common_nodes, common_nodes].astype(float)

    # 对角线设为 NaN（不参与评估）
    np.fill_diagonal(gt.values, np.nan)
    np.fill_diagonal(mono.values, np.nan)

    # 预测矩阵：P_mono >= alpha 视为正类（支持单调性），否则负类
    pred = (mono >= alpha).astype(int)

    # 构建对比标签矩阵
    # TP: gt=1, pred=1
    # TN: gt=0, pred=0
    # FP: gt=0, pred=1
    # FN: gt=1, pred=0
    comparison = pd.DataFrame(index=common_nodes, columns=common_nodes, dtype=str)
    for u in common_nodes:
        for v in common_nodes:
            if u == v:
                comparison.loc[u, v] = ''
                continue
            g = gt.loc[u, v]
            p = pred.loc[u, v]
            if pd.isna(g) or pd.isna(p):
                comparison.loc[u, v] = ''
            elif g == 1 and p == 1:
                comparison.loc[u, v] = 'TP'
            elif g == 0 and p == 0:
                comparison.loc[u, v] = 'TN'
            elif g == 0 and p == 1:
                comparison.loc[u, v] = 'FP'
            elif g == 1 and p == 0:
                comparison.loc[u, v] = 'FN'

    # 扁平化计算统计量
    gt_flat = gt.values.flatten()
    pred_flat = pred.values.flatten()
    # 移除 NaN（对角线）
    mask = ~np.isnan(gt_flat)
    gt_flat = gt_flat[mask].astype(int)
    pred_flat = pred_flat[mask].astype(int)

    tp = np.sum((gt_flat == 1) & (pred_flat == 1))
    tn = np.sum((gt_flat == 0) & (pred_flat == 0))
    fp = np.sum((gt_flat == 0) & (pred_flat == 1))
    fn = np.sum((gt_flat == 1) & (pred_flat == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

    if verbose:
        print("=" * 60)
        print(f"Evaluation Results (alpha = {alpha})")
        print("=" * 60)
        print(f"True Positives  (edges correctly supported): {tp}")
        print(f"True Negatives  (non-edges correctly unsupported): {tn}")
        print(f"False Positives (non-edges wrongly supported): {fp}")
        print(f"False Negatives (edges wrongly unsupported): {fn}")
        print("-" * 40)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print("=" * 60)

    return {
        'ground_truth': gt,
        'monotonicity': mono,
        'predicted': pred,
        'comparison': comparison,
        'metrics': metrics
    }


def __print_comparison_matrix(comparison_df: pd.DataFrame):
    """以更友好的格式打印对比矩阵"""
    # 替换为空字符串方便阅读
    display_df = comparison_df.replace('', np.nan)
    print("\nComparison Matrix (TP/TN/FP/FN):")
    print(display_df.to_string())


def generate_monotonicity_blacklist(
        csv_path: str,
        alpha: float = 0.1,
        topo_source: Optional[str] = None
):
    """
    从 CSV 样本数据计算单调性后验概率，生成禁止边列表并保存为 JSON（含所有边的 P_mono）。

    参数
    ----------
    csv_path : str
        观测样本 CSV 文件的完整路径。
    alpha : float, default=0.1
        禁止边阈值（P_mono < alpha）。
    topo_source : str, optional
        关联的拓扑源文件名，用于元数据记录。

    返回
    -------
    List[Tuple[str, str]]
        禁止边列表，格式 [(parent, child), ...]，可直接用于 pgmpy 的 black_list。
    """
    # 1. 调用批量单调性检验
    result = batch_monotonicity_test_from_csv(csv_path, verbose=False)
    pairs_df = result['pairs']

    # 2. 筛选禁止边
    forbidden_df = pairs_df[pairs_df['P_mono'] < alpha][['Parent', 'Child', 'P_mono']]
    forbidden_list = [(row.Parent, row.Child) for row in forbidden_df.itertuples(index=False)]

    # 2.5 新增，计算惩罚项
    forbidden_penality = generate_monotonicity_penalties(forbidden_df, alpha)

    # 3. 准备所有边及禁止边的 JSON 数据
    all_pairs_json = [
        {"from": row.Parent, "to": row.Child, "P_mono": round(row.P_mono, 6)}
        for row in pairs_df.itertuples(index=False)
    ]

    forbidden_edges_json = [
        {"from": row.Parent, "to": row.Child, "penality": round(forbidden_penality[(row.Parent, row.Child)], 6)}
        for row in forbidden_df.itertuples(index=False)
    ]

    # 4. 构建输出路径（例如 S1000.csv -> S1000_mono_blacklist_alpha0d1.json）
    dir_name = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    alpha_str = str(alpha).replace('.', 'd')
    output_filename = f"{base_name}_mono_blacklist_alpha{alpha_str}.json"
    output_path = os.path.join(dir_name, output_filename)

    # 5. 输出 JSON
    output_data = {
        "metadata": {
            "description": "Forbidden edges derived from Bayesian weak monotonicity test, with all P_mono values",
            "alpha": alpha,
            "total_ordered_pairs": len(pairs_df),
            "forbidden_edges_count": len(forbidden_list),
            "source_csv": os.path.basename(csv_path),
            "source_topo": topo_source,
            "generated_at": datetime.now().isoformat()
        },
        "all_pairs_pmono": all_pairs_json,
        "forbidden_edges_by_mono": forbidden_edges_json
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Monotonicity blacklist saved to: {output_path}")
    print(f"  - Forbidden edges: {len(forbidden_list)} / {len(pairs_df)}")

    return forbidden_list


def generate_monotonicity_penalties(pairs_df, alpha):
    """返回 { (parent, child) : max(0, alpha - P_mono) } 未乘系数 beta"""
    pen = {}
    for _, row in pairs_df.iterrows():
        u, v, p = row['Parent'], row['Child'], row['P_mono']
        if p < alpha:
            pen[(u, v)] = alpha - p
    return pen

# # ================================================================
# # 主测试程序
# # ================================================================
# if __name__ == "__main__":
#     # 配置路径
#     TOPO_JSON = "topo.json"
#     SAMPLE_CSV = 'generated_samples.csv'
#
#     generate_monotonicity_blacklist(SAMPLE_CSV)
#     exit(0)
#
#     # 加载 ground truth
#     print(f"Loading ground truth from '{TOPO_JSON}'...")
#     gt_matrix = __load_ground_truth_from_json(TOPO_JSON)
#     print(f"Ground truth loaded: {len(gt_matrix)} nodes, {gt_matrix.sum().sum():.0f} edges.")
#
#     result = batch_monotonicity_test_from_csv(csv_path=SAMPLE_CSV)
#     mono_matrix = result['matrix']
#
#     # 对齐节点顺序（按 ground truth 节点列表）
#     nodes = gt_matrix.columns.tolist()
#     mono_matrix = mono_matrix.reindex(index=nodes, columns=nodes)
#
#     # 进行评估
#     alpha = 0.80  # 可调整阈值
#     result = __evaluate_monotonicity(gt_matrix, mono_matrix, alpha=alpha, verbose=True)
#
#     # 打印部分对比矩阵（若节点过多，可仅显示前几行几列）
#     __print_comparison_matrix(result['comparison'].iloc[:10, :10])
#
#     # 可选：保存结果
#     result['comparison'].to_csv("comparison_matrix.csv")
#     pd.DataFrame([result['metrics']]).to_csv("evaluation_metrics.csv", index=False)
#     print("\nResults saved to 'comparison_matrix.csv' and 'evaluation_metrics.csv'.")
