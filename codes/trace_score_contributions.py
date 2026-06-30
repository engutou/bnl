#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trace_score_contributions.py
在爬山法搜索过程中，记录每一步的 BDeu 得分变化和单调性惩罚变化。
独立运行，仅对 mono_score 算法，使用指定的 BN 实例和样本量。
"""

import json
import os
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from pgmpy.estimators import BDeu, HillClimbSearch
from pgmpy.estimators.StructureScore import BDeu as BaseBDeu
from pgmpy.base import DAG

# ===================== 配置 =====================
BASE_DIR = "../experiments/generated_bns"
BN_INSTANCE = "bn_N24_E35_01"  # 选择要分析的 BN 实例
SAMPLE_SIZE = "S100"  # 选择样本量
ALPHA = 0.3  # 单调性阈值
BETA = 100  # 先验精度
ESS = 1  # BDeu 等效样本大小
OUTPUT_CSV = "score_trace.csv"  # 输出文件名


# ===================== 自定义评分类（带得分追踪）=====================
class TraceableMonoPenaltyBDeu(BaseBDeu):
    """
    带有单调性惩罚项的 BDeu 评分函数，同时记录每一步的得分分解。
    """

    def __init__(self, data, beta=500, expert_knowledge=None,
                 equivalent_sample_size=1.0, **kwargs):
        super().__init__(data, equivalent_sample_size=equivalent_sample_size, **kwargs)
        self.beta = beta
        self.expert_knowledge = expert_knowledge or []
        # 追踪记录
        self.trace = []  # 每条记录: {iteration, node, parents, bdeu_score, penalty, total_score}

    def local_score(self, variable, parents):
        base = super().local_score(variable, parents)
        penalty = 0.0
        for from_v, to_v, pen in self.expert_knowledge:
            if to_v == variable and from_v in parents:
                penalty += pen
        total = base - self.beta * penalty
        # 记录本次调用
        self.trace.append({
            'variable': variable,
            'parents': str(list(parents)),
            'bdeu_score': round(base, 4),
            'penalty': round(penalty, 6),
            'total_score': round(total, 4)
        })
        return total


# ===================== 数据加载 =====================
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_data_files(bn_dir, sample_size):
    """获取指定 BN 实例和样本量的数据文件路径"""
    data_dir = bn_dir / "data" / sample_size
    csv_files = list(data_dir.glob("*.csv"))
    mono_files = list(data_dir.glob("*_mono_blacklist_alpha*.json"))
    hier_file = bn_dir / "ground_truth" / "topo_hier_blacklist.json"

    if not csv_files or not mono_files or not hier_file.is_file():
        raise FileNotFoundError(f"Missing files for {bn_dir.name}/{sample_size}")

    return str(csv_files[0]), str(mono_files[0]), str(hier_file)


def load_expert_knowledge(hier_path, mono_path, alpha):
    """加载层级黑名单和单调性惩罚三元组"""
    hier_data = load_json(hier_path)
    mono_data = load_json(mono_path)

    # 层级黑名单
    forbidden_hier = [(h["from"], h["to"]) for h in hier_data["forbidden_edges_by_hier"]]

    # 单调性惩罚三元组
    penalty_triplets = []
    if "all_pairs_pmono" in mono_data:
        for pair in mono_data["all_pairs_pmono"]:
            u, v, p = pair["from"], pair["to"], pair["P_mono"]
            if p < alpha:
                penalty_triplets.append((u, v, alpha - p))

    return forbidden_hier, penalty_triplets


# ===================== 主程序 =====================
def main():
    root = Path(BASE_DIR)
    bn_dir = root / BN_INSTANCE

    if not bn_dir.is_dir():
        print(f"BN instance {BN_INSTANCE} not found in {BASE_DIR}")
        return

    # 获取数据文件
    csv_path, mono_path, hier_path = get_data_files(bn_dir, SAMPLE_SIZE)
    data = pd.read_csv(csv_path, sep=',')
    print(f"Loaded data: {data.shape[0]} samples, {data.shape[1]} variables")

    # 加载专家知识
    forbidden_hier, penalty_triplets = load_expert_knowledge(hier_path, mono_path, ALPHA)
    print(f"Hierarchical blacklist: {len(forbidden_hier)} edges")
    print(f"Monotonicity penalty triplets: {len(penalty_triplets)} edges")
    print(f"  Average penalty value: {np.mean([p for _, _, p in penalty_triplets]):.4f}")

    # 创建带追踪的评分对象
    scorer = TraceableMonoPenaltyBDeu(
        data,
        beta=BETA,
        expert_knowledge=penalty_triplets,
        equivalent_sample_size=ESS
    )

    # 准备 ExpertKnowledge（层级黑名单）
    from pgmpy.estimators import ExpertKnowledge
    ek = ExpertKnowledge()
    ek.forbidden_edges = forbidden_hier

    # 执行爬山搜索
    print(f"\nStarting hill-climbing search with mono_score (alpha={ALPHA}, beta={BETA}, ESS={ESS})...")
    t0 = time.time()
    hc = HillClimbSearch(data)
    best_model = hc.estimate(
        scoring_method=scorer,
        expert_knowledge=ek,
        show_progress=False
    )
    elapsed = time.time() - t0
    print(f"Search completed in {elapsed:.1f} seconds")
    print(f"Learned edges: {len(best_model.edges())}")

    # 保存追踪数据
    trace_df = pd.DataFrame(scorer.trace)
    trace_df.index.name = 'iteration'
    trace_df.to_csv(OUTPUT_CSV, index=True)
    print(f"\nScore trace saved to {OUTPUT_CSV} ({len(trace_df)} rows)")

    # 统计摘要
    print("\n===== Score Trace Summary =====")
    print(f"Total local_score calls: {len(trace_df)}")
    print(f"BDeu score range: [{trace_df['bdeu_score'].min():.2f}, {trace_df['bdeu_score'].max():.2f}]")
    print(f"Penalty range: [{trace_df['penalty'].min():.6f}, {trace_df['penalty'].max():.6f}]")
    print(f"Total score range: [{trace_df['total_score'].min():.2f}, {trace_df['total_score'].max():.2f}]")
    print(f"\nBDeu score statistics:")
    print(f"  Mean: {trace_df['bdeu_score'].mean():.2f}")
    print(f"  Std: {trace_df['bdeu_score'].std():.2f}")
    print(f"  Median: {trace_df['bdeu_score'].median():.2f}")
    print(f"\nPenalty statistics (non-zero only):")
    non_zero = trace_df[trace_df['penalty'] > 0]['penalty']
    if len(non_zero) > 0:
        print(f"  Count: {len(non_zero)}")
        print(f"  Mean: {non_zero.mean():.6f}")
        print(f"  Max: {non_zero.max():.6f}")
    else:
        print("  No non-zero penalties recorded")


if __name__ == "__main__":
    main()