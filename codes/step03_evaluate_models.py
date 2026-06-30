#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step03_evaluate_models.py
评估学习到的贝叶斯网络结构与真实结构 (ground truth) 的差异。
输出:
  - 每个学习结果同目录下的 *_evaluation.json (单次评估)
  - experiments/evaluation_summary.csv (全局汇总)
"""

import json
import os
import glob
import re
import csv
import numpy as np
import networkx as nx
from collections import defaultdict


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------

def load_json(filepath):
    """加载 JSON 文件并返回 Python 对象。"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_edges_from_json(json_data):
    """
    从贝叶斯网络 JSON 结构中提取有向边集合。
    JSON 格式: {"edges": {"single_edges": [{"from": "...", "to": "..."}, ...]}}
    返回: set of tuples (from, to)
    """
    edges = json_data.get('edges', {}).get('single_edges', [])
    return {(e['from'], e['to']) for e in edges}


def extract_undirected_edges(edges):
    """
    从可能包含双向边的边集合中检测无向边。
    CPDAG 通过双向边表示无向边，即同时存在 (u, v) 和 (v, u)。
    返回:
        undirected: set of frozenset 表示无向边
        directed: set of tuples 表示真正的有向边（排除了双向）
    """
    undirected = set()
    directed = set()
    processed = set()
    for u, v in edges:
        if (v, u) in edges:
            if (v, u) not in processed and (u, v) not in processed:
                # 双向边，视为无向边
                undirected.add(frozenset([u, v]))
                processed.add((u, v))
                processed.add((v, u))
        else:
            directed.add((u, v))
    return undirected, directed


def compute_skeleton_metrics(true_edges, learned_edges):
    """
    计算骨架级别的指标（忽略方向，仅考虑边的存在性）。
    参数:
        true_edges: set of tuples (from, to) 真实有向边
        learned_edges: set of tuples 学习到的边（可能包含双向边表示无向）
    返回:
        dict 包含:
            skeleton_TP: 正确识别的无向边数量
            skeleton_FP: 多出的无向边数量
            skeleton_FN: 遗漏的真实无向边数量
            skeleton_SHD: 骨架汉明距离 (FP + FN)
    """
    # 提取真实骨架（仅节点对，忽略方向）
    true_skeleton = {frozenset([u, v]) for u, v in true_edges}
    # 从学习边提取骨架（无论有向或无向）
    learned_skeleton = set()
    for u, v in learned_edges:
        learned_skeleton.add(frozenset([u, v]))

    tp = len(true_skeleton & learned_skeleton)
    fp = len(learned_skeleton - true_skeleton)
    fn = len(true_skeleton - learned_skeleton)
    shd = fp + fn
    return {
        'skeleton_TP': tp,
        'skeleton_FP': fp,
        'skeleton_FN': fn,
        'skeleton_SHD': shd
    }


def compute_directed_metrics(true_edges, learned_edges):
    """
    计算考虑方向的指标。
    处理 CPDAG 的无向边：对于无向边，不将其作为定向正确或错误，而是统计为“方向未知”。
    返回:
        dict 包含:
            TP: 方向正确的边数
            FP: 方向错误或多余的边数（不包含无向边中的反向）
            FN: 缺失或方向错误（不匹配）的真实边数
            SHD: 汉明距离（考虑方向）
            precision, recall, F1
            reversed_pairs: 方向反转的边集合
            undirected_in_learned: 学习结果中的无向边数
    """
    # 分离无向边和有向边
    undirected, directed = extract_undirected_edges(learned_edges)

    # 真实边集合
    true_set = set(true_edges)
    # 学习到的有向边（仅方向确定的）
    learned_directed_set = directed

    # 正确识别且方向一致
    tp = len(true_set & learned_directed_set)
    # 多出的有向边（方向错误的也算在这里，但暂不考虑反向边调整）
    fp = len(learned_directed_set - true_set)
    # 缺失的真实边（包括方向错误或完全缺失）
    # 首先找出真实边中哪些没有在 learned_directed_set 中
    fn_total = len(true_set - learned_directed_set)

    # 更精细的分解：检查反向边（即学到的方向恰好相反）
    reversed_edges = set()
    for u, v in true_set:
        if (v, u) in learned_directed_set:
            reversed_edges.add((u, v))
    # 在标准 SHD 计算中，一条反向边计为一次“反转”操作，相当于 1 次编辑距离，
    # 且同时贡献 FP 和 FN。为与 pgmpy SHD 定义一致，我们直接计算 SHD = FP + FN。
    # 但上面的 FP/FN 没有扣除重叠部分，这里直接用集合运算修正：
    # SHD = |E_true \ E_learned| + |E_learned \ E_true|
    # 注意：无向边在 SHD 中如何处理？一般对 CPDAG 需要选择最优定向。
    # 简化处理：如果存在无向边，暂时不纳入 directed metrics，或记录其数量。

    # 为准确，我们计算不考虑无向边的 SHD：只比较有向边
    # 但有向边中反向边会被算两次（一次 FN 一次 FP），实际上反向边只需一次编辑（反转），
    # 这需要调整。这里我们遵循标准 SHD 定义：对两条有向边，若方向相反算作 1 次编辑。
    # 因此，我们直接计算 SHD 如下：

    # 将所有有向边（包括真实和学习）统一计算
    shd = len(true_set - learned_directed_set) + len(learned_directed_set - true_set)
    # 但上述计算中，反向边对会贡献 2，实际上只应贡献 1。所以减去反向边的数量：
    # 对于每对反向边，它在 true_set - learned_set 和 learned_set - true_set 中各出现一次，
    # 因此减去 len(reversed_edges) 即可。
    shd -= len(reversed_edges)

    # 重新计算 TP、FP、FN 基于调整后的集合
    # 其实可以直接用 shd 的计算，但为了保留可解释的 TP/FP/FN，我们采用：
    # 去掉反向边的影响：定义 adjusted_learned = (learned_directed_set - reversed反向边) ∪ corrected方向
    # 较复杂。这里采用简化：保留原始 TP/FP，只记录 reversed。

    # 计算 precision/recall/f1 仅针对有向边
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn_total) if (tp + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn_total,
        'SHD': shd,
        'directed_precision': precision,
        'directed_recall': recall,
        'directed_F1': f1,
        'num_reversed': len(reversed_edges),
        'reversed_edges': list(reversed_edges),
        'num_undirected': len(undirected),
        'undirected_edges': [list(e) for e in undirected]  # 转为可序列化
    }


def compute_full_metrics(true_edges, learned_edges):
    """
    综合评估：骨架指标、有向指标、SHD。
    返回一个字典，包含所有指标。
    """
    # 骨架
    skeleton = compute_skeleton_metrics(true_edges, learned_edges)
    # 有向
    directed = compute_directed_metrics(true_edges, learned_edges)
    # 合并
    metrics = {**skeleton, **directed}
    # 学习到的总边数
    metrics['learned_edges_count'] = len(learned_edges)
    metrics['true_edges_count'] = len(true_edges)
    return metrics


# ----------------------------------------------------------------------
# 主评估流程
# ----------------------------------------------------------------------

def evaluate_all(strategy_filter=None):
    """
    遍历所有 BN 实例、策略、样本量，执行评估。
    strategy_filter: 可选，包含需评估策略名称的列表（如 ['HC_BDeu', 'hierarchy_mono_score']）。
                     若为 None，则评估所有策略。
    """
    exp_dir = 'experiments_N12_E16'
    base_dir = f"../{exp_dir}/generated_bns"
    summary_csv_path = f"../{exp_dir}/evaluation_summary-v3.csv"

    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)

    all_rows = []

    for bn_dir in sorted(glob.glob(os.path.join(base_dir, "bn_*"))):
        bn_name = os.path.basename(bn_dir)
        gt_file = os.path.join(bn_dir, "ground_truth", "dag.json")
        if not os.path.exists(gt_file):
            print(f"[警告] {bn_name} 缺少 ground truth，跳过")
            continue
        gt_data = load_json(gt_file)
        gt_edges = extract_edges_from_json(gt_data)
        print(f"\n处理实例 {bn_name}: 真实边数 {len(gt_edges)}")

        results_dir = os.path.join(bn_dir, "results")
        if not os.path.isdir(results_dir):
            print(f"  无 results 目录，跳过")
            continue

        for strategy_dir in sorted(glob.glob(os.path.join(results_dir, "*"))):
            strategy = os.path.basename(strategy_dir)
            # 如果指定了过滤列表，且当前策略不在列表中，则跳过
            if strategy_filter is not None and not any(strategy.startswith(s) for s in strategy_filter):
                continue

            # if 'alpha' in strategy:
            #     continue

            beta_match = re.search(r'_beta(\d+)', strategy)
            beta_val = int(beta_match.group(1)) if beta_match else None

            for sample_dir in sorted(glob.glob(os.path.join(strategy_dir, "S*"))):
                sample_size_str = os.path.basename(sample_dir)
                # 提取数字部分
                sample_match = re.search(r'\d+', sample_size_str)
                sample_size = int(sample_match.group()) if sample_match else 0

                # 查找学习到的结构 JSON 文件
                json_files = glob.glob(os.path.join(sample_dir, "*_learned_edges.json"))
                if not json_files:
                    print(f"  跳过 {strategy}/{sample_size_str} (无 learned_edges.json)")
                    continue
                learned_file = json_files[0]  # 假设只有一个

                # 加载学习到的边
                print(learned_file)
                learned_data = load_json(learned_file)
                learned_edges = extract_edges_from_json(learned_data)

                # 计算指标
                metrics = compute_full_metrics(gt_edges, learned_edges)

                # 构建元数据
                metadata = {
                    "bn_instance": bn_name,
                    "strategy": strategy,
                    "beta": beta_val,
                    "sample_size": sample_size,
                    "num_samples": learned_data.get("metadata", {}).get("data_shape", [0])[
                        0] if "data_shape" in learned_data.get("metadata", {}) else sample_size,
                    "is_cpdag": metrics['num_undirected'] > 0
                }

                # 保存单结果 JSON
                eval_result = {
                    "metadata": metadata,
                    "metrics": metrics
                }
                eval_file = os.path.join(sample_dir, f"{sample_size_str}_evaluation.json")
                with open(eval_file, 'w', encoding='utf-8') as f:
                    json.dump(eval_result, f, indent=2, ensure_ascii=False)

                # 添加一行汇总数据
                row = {
                    **metadata,
                    **metrics
                }
                all_rows.append(row)
                print(f"  ✓ {strategy}/{sample_size_str} -> SHD={metrics['SHD']}, F1={metrics['directed_F1']:.3f}")

    # 保存全局汇总 CSV
    if all_rows:
        # 确定所有列名
        fieldnames = list(all_rows[0].keys())
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n全局汇总已保存至 {summary_csv_path}，共 {len(all_rows)} 行。")
    else:
        print("没有生成任何评估结果。")


if __name__ == "__main__":
    # 只评估指定策略，若想评估所有策略可设为 None 或不传参
    # selected_strategies = ["PC", 'GES', "HC_BDeu"]
    selected_strategies = [
        "PC", 'GES', "HC_BDeu",
        "hierarchy",
        "monotonic_before",
        "monotonic_after",
        "monotonic_score",
        "hierarchy_monotonic_before",
        "hierarchy_monotonic_after",
        "hierarchy_monotonic_score"
    ]
    evaluate_all(strategy_filter=selected_strategies)