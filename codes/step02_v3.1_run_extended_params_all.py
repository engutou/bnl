#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_extended_params_all.py
针对全部 6 个 monotonic 相关策略，使用扩展的 alpha 和 beta 值运行补充实验。
跳过已存在的实验结果，只执行新的参数组合。
结果可直接用于 step03_evaluate_models.py 评估。
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
from pgmpy.estimators import BDeu, ExpertKnowledge, HillClimbSearch
from pgmpy.estimators.StructureScore import BDeu as BaseBDeu

import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)


# =========================================================================
# 自定义评分函数（与 step02 完全一致）
# =========================================================================
class MonoPenaltyBDeu(BaseBDeu):
    """带单调性惩罚的 BDeu 评分函数（方案B）"""
    def __init__(self, data, beta=500, expert_knowledge=None,
                 equivalent_sample_size=10.0, **kwargs):
        super().__init__(data, equivalent_sample_size=equivalent_sample_size, **kwargs)
        self.beta = beta
        self.expert_knowledge = expert_knowledge or []

    def local_score(self, variable, parents):
        base = super().local_score(variable, parents)
        penalty = 0.0
        for from_v, to_v, pen in self.expert_knowledge:
            if to_v == variable and from_v in parents:
                penalty += pen
        return base - self.beta * penalty


# =========================================================================
# 工具函数
# =========================================================================
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ek_json_data(hier_blacklist_path_: str, mono_blacklist_path_: str,
                      alpha_override: Optional[float] = None) -> Dict:
    """加载层级黑名单和单调性约束，支持动态重算惩罚。"""
    with open(hier_blacklist_path_, 'r', encoding='utf-8') as f:
        hier_data = json.load(f)
    with open(mono_blacklist_path_, 'r', encoding='utf-8') as f:
        mono_data = json.load(f)

    forbidden_hier = [(h["from"], h["to"]) for h in hier_data["forbidden_edges_by_hier"]]
    alpha = alpha_override if alpha_override is not None else mono_data.get("metadata", {}).get("alpha", 0.1)

    forbidden_mono = []
    penalty_triplets = []
    pmono_dict = {}

    if "all_pairs_pmono" in mono_data:
        for pair in mono_data["all_pairs_pmono"]:
            u, v, p = pair["from"], pair["to"], pair["P_mono"]
            pmono_dict[(u, v)] = p
            if p < alpha:
                forbidden_mono.append((u, v))
                penalty_triplets.append((u, v, alpha - p))

    return {
        "forbidden_edges_by_hier": forbidden_hier,
        "forbidden_edges_by_mono": forbidden_mono,
        "penalty_triplets": penalty_triplets,
        "pmono_dict": pmono_dict,
        "alpha": alpha
    }


def get_data_json(root_dir: str) -> dict:
    """扫描目录，获取所有 BN 实例的数据路径。"""
    root = Path(root_dir)
    result = {}
    for bn_dir in root.glob("bn_*"):
        if not bn_dir.is_dir():
            continue
        bn_name = bn_dir.name
        data_dir = bn_dir / "data"
        if not data_dir.is_dir():
            continue
        sample_mono = []
        s_dirs = sorted(
            [d for d in data_dir.glob("S*") if d.is_dir() and re.match(r'S\d+$', d.name, re.I)],
            key=lambda d: int(re.search(r'\d+', d.name).group())
        )
        for s_dir in s_dirs:
            csv_files = list(s_dir.glob("*.csv"))
            if not csv_files:
                continue
            mono_files = list(s_dir.glob("*_mono_blacklist_alpha*.json"))
            if not mono_files:
                continue
            sample_mono.append([f"./{csv_files[0].as_posix()}",
                                f"./{mono_files[0].as_posix()}"])
        hier_file = bn_dir / "ground_truth" / "topo_hier_blacklist.json"
        if not hier_file.is_file():
            continue
        result[bn_name] = {
            "sample_mono": sample_mono,
            "hier_blacklist": f"./{hier_file.as_posix()}"
        }
    return result


def process_save_path(orig_csv_path, strategy_tag):
    """根据样本路径和策略标签生成结果保存目录。"""
    dir_path = os.path.dirname(orig_csv_path)
    dir_path = dir_path.replace("data", f"results/{strategy_tag}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def result_exists(sample_csv_path, tag):
    """检查对应结果目录下是否已有 learned_edges.json 文件。"""
    res_dir = process_save_path(sample_csv_path, tag)
    json_files = list(Path(res_dir).glob("*_learned_edges.json"))
    return len(json_files) > 0


def save_learned_structure(model, data, output_dir, tag):
    """保存学习到的 DAG 为 JSON 文件。"""
    node_info = {node: [str(v) for v in sorted(data[node].dropna().unique())]
                 for node in model.nodes()}
    json_structure = {
        "metadata": {
            "learning_method": tag,
            "data_shape": list(data.shape)
        },
        "individual_variables": node_info,
        "edges": {
            "single_edges": [{"from": e[0], "to": e[1]} for e in model.edges()]
        }
    }
    file_pre = os.path.basename(output_dir.rstrip('/'))
    json_path = os.path.join(output_dir, f"{file_pre}_learned_edges.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_structure, f, ensure_ascii=False, indent=2)
    return json_path


# =========================================================================
# 主程序
# =========================================================================
if __name__ == '__main__':
    base_path = "../experiments_N12_E16/generated_bns/"

    # 全部 6 个 alpha 敏感策略
    strategies = [
        "monotonic_before",
        "monotonic_after",
        "monotonic_score",
        # "hierarchy_monotonic_before",
        # "hierarchy_monotonic_after",
        # "hierarchy_monotonic_score"
    ]

    # 原有的 + 扩展的超参数
    ALL_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ALL_BETAS = [10, 30, 50, 70, 100, 200, 500, 1000]
    ESS_VALUES = [1.0, 10.0]

    # 创建标签：alpha 中的小数点替换为 'd'
    def alpha_to_str(a):
        return str(a).replace('.', 'd')

    data_json_dict = get_data_json(base_path)
    total_jobs = 0
    skipped_jobs = 0

    for bn_name, bn_info in data_json_dict.items():
        for sample_csv_path, mono_blacklist_path in bn_info["sample_mono"]:
            for strategy in strategies:
                is_score = 'score' in strategy
                for ess_val in ESS_VALUES:
                    for alpha_val in ALL_ALPHAS:
                        # 确定 beta 列表：非 score 固定 500
                        betas = ALL_BETAS if is_score else [500]
                        for beta_val in betas:
                            # 构造标签
                            if is_score:
                                tag = f"{strategy}_ess{int(ess_val)}_alpha{alpha_to_str(alpha_val)}_beta{beta_val}"
                            else:
                                # 非 score 策略不加 beta 后缀（但 beta 固定 500，我们可省略）
                                tag = f"{strategy}_ess{int(ess_val)}_alpha{alpha_to_str(alpha_val)}"

                            # 跳过已存在的组合
                            if result_exists(sample_csv_path, tag):
                                skipped_jobs += 1
                                continue

                            print("*" * 24, f"{sample_csv_path} {tag}", "*" * 24)

                            # 加载专家知识（动态重算 alpha）
                            ek = load_ek_json_data(
                                bn_info["hier_blacklist"],
                                mono_blacklist_path,
                                alpha_override=alpha_val
                            )

                            # 读取数据
                            data = pd.read_csv(sample_csv_path, sep=',')

                            # 准备评分函数和专家知识对象
                            if is_score:
                                scorer = MonoPenaltyBDeu(
                                    data,
                                    beta=beta_val,
                                    expert_knowledge=ek.get('penalty_triplets', []),
                                    equivalent_sample_size=ess_val
                                )
                            else:
                                # 非 score 策略使用普通 BDeu（惩罚仅通过黑名单或后处理实现）
                                # 此处我们根据策略选择：before 对应黑名单，after 对应后处理。
                                # 但为了统一，这里只需创建普通 BDeu 评分对象，
                                # 实际的单调性约束在搜索前或搜索后通过黑名单或剪枝实现。
                                # 为了代码简洁，我们将所有爬山法都使用 BDeu(ess)，
                                # 然后在 learn_structure 中根据策略施加单调性约束。
                                scorer = BDeu(data, equivalent_sample_size=ess_val)

                            # 准备 ExpertKnowledge（层级黑名单）
                            ek_obj = ExpertKnowledge()
                            ek_obj.forbidden_edges = ek.get('forbidden_edges_by_hier', [])

                            # 处理单调性约束（非 score 策略）
                            if strategy == "monotonic_before":
                                ek_obj.forbidden_edges = list(set(
                                    ek_obj.forbidden_edges + ek.get('forbidden_edges_by_mono', [])
                                ))
                            elif strategy == "hierarchy_monotonic_before":
                                combined = list(set(
                                    ek_obj.forbidden_edges + ek.get('forbidden_edges_by_mono', [])
                                ))
                                ek_obj.forbidden_edges = combined
                            # after 策略：先搜索，后剪枝。我们在搜索后进行处理。
                            post_prune = (strategy in ["monotonic_after", "hierarchy_monotonic_after"])

                            # 爬山搜索
                            hc = HillClimbSearch(data)
                            best_model = hc.estimate(
                                scoring_method=scorer,
                                expert_knowledge=ek_obj,
                                show_progress=False
                            )

                            # 后处理剪枝（after 策略）
                            if post_prune:
                                pmono_dict = ek.get('pmono_dict', {})
                                alpha = ek.get('alpha', 0.1)
                                to_remove = []
                                for u, v in best_model.edges():
                                    if pmono_dict.get((u, v), 1.0) < alpha:
                                        to_remove.append((u, v))
                                for u, v in to_remove:
                                    best_model.remove_edge(u, v)
                                print(f"  Post-hoc pruning removed {len(to_remove)} edges")

                            # 保存结果
                            save_dir = process_save_path(sample_csv_path, tag)
                            save_learned_structure(best_model, data, save_dir, tag)
                            total_jobs += 1
                            print(f"  ✓ Edges={len(best_model.edges())}, saved to {save_dir}")

    print(f"\nAll tasks finished. New jobs: {total_jobs}, skipped: {skipped_jobs}.")