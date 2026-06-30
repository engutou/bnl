#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Not_used_step02_build_model_from_samples.py
从观测样本数据学习贝叶斯网络结构，支持多种知识融合策略。
重构后支持 ESS, alpha, beta 超参数扫描。
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pgmpy.estimators import BDeu, ExpertKnowledge, HillClimbSearch, GES, MmhcEstimator, PC
from pgmpy.estimators.StructureScore import BDeu as BaseBDeu

# 屏蔽pgmpy的冗余日志
logging.getLogger("pgmpy").setLevel(logging.WARNING)


# =========================================================================
# 自定义带单调性惩罚的 BDeu 评分类（方案B）
# =========================================================================
class MonoPenaltyBDeu(BaseBDeu):
    """带单调性结构先验的 BDeu 评分函数（方案B）。
    总得分 = 原始 BDeu 对数边际似然 - beta * sum(max(0, alpha - P_mono))
    expert_knowledge 格式：[(from, to, penalty), ...]  其中 penalty = max(0, alpha - P_mono)
    """

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
# 贝叶斯网络学习器
# =========================================================================
class BayesianNetworkLearner:
    def __init__(self, data, variable_names=None):
        if isinstance(data, pd.DataFrame):
            self.data = data
            self.variable_names = list(data.columns)
        elif isinstance(data, np.ndarray):
            if variable_names is None:
                raise ValueError("对于numpy数组，必须提供variable_names")
            self.data = pd.DataFrame(data, columns=variable_names)
            self.variable_names = variable_names
        else:
            raise TypeError("data必须是pandas DataFrame或numpy数组")
        print(f"数据形状: {self.data.shape}, 变量 {len(self.variable_names)} 个")

    # ------------------------------------------------------------------
    # 爬山法结构学习 (支持所有策略和超参数)
    # ------------------------------------------------------------------
    def learn_structure_hill_climbing(self,
                                      strategy,
                                      max_indegree=None,
                                      show_progress=True,
                                      epsilon=1e-4,
                                      max_iter=1000,
                                      equivalent_sample_size=10.0,
                                      ek=None,
                                      beta=500):
        print("开始结构学习（爬山算法）...")

        # 默认使用 BDeu(ESS)
        scoring_method_obj = BDeu(self.data, equivalent_sample_size=equivalent_sample_size)
        expert_knowledge = ExpertKnowledge() if ek is not None else None
        post_prune_mono = False
        alpha = ek.get('alpha', 0.1) if ek else 0.1

        if ek is not None:
            # 层级黑名单
            if strategy in ("hierarchy", "hierarchy_monotonic_before",
                            "hierarchy_monotonic_after", "hierarchy_monotonic_score",
                            "monotonic_after"):
                expert_knowledge.forbidden_edges = ek['forbidden_edges_by_hier']

            if strategy == "monotonic_before":
                expert_knowledge.forbidden_edges = ek['forbidden_edges_by_mono']

            elif strategy == "hierarchy_monotonic_before":
                combined = list(set(ek['forbidden_edges_by_hier'] + ek['forbidden_edges_by_mono']))
                expert_knowledge.forbidden_edges = combined

            elif strategy in ("monotonic_after", "hierarchy_monotonic_after"):
                expert_knowledge.forbidden_edges = ek.get('forbidden_edges_by_hier', [])
                post_prune_mono = True

            elif strategy in ("monotonic_score", "hierarchy_monotonic_score"):
                scoring_method_obj = MonoPenaltyBDeu(
                    self.data,
                    beta=beta,
                    equivalent_sample_size=equivalent_sample_size,
                    expert_knowledge=ek.get('penalty_triplets', [])
                )
                if strategy == "hierarchy_monotonic_score":
                    expert_knowledge.forbidden_edges = ek.get('forbidden_edges_by_hier', [])
                else:
                    expert_knowledge = ExpertKnowledge()
            # hierarchy 或 HC_BDeu: 只使用层级黑名单或无黑名单
        else:
            expert_knowledge = None

        hc = HillClimbSearch(self.data)
        best_model = hc.estimate(
            scoring_method=scoring_method_obj,
            max_indegree=max_indegree,
            epsilon=epsilon,
            max_iter=max_iter,
            show_progress=show_progress,
            expert_knowledge=expert_knowledge
        )

        # 后处理剪枝（方案A）
        if post_prune_mono and ek and 'pmono_dict' in ek:
            print("执行后处理单调性剪枝 (方案A)...")
            pmono_dict = ek['pmono_dict']
            to_remove = []
            for u, v in best_model.edges():
                p = pmono_dict.get((u, v), 1.0)
                if p < alpha:
                    to_remove.append((u, v))
            for u, v in to_remove:
                best_model.remove_edge(u, v)
            print(f"  剪枝移除 {len(to_remove)} 条边")

        print(f"✓ HC完成, 边数: {len(best_model.edges())}")
        return best_model

    # ------------------------------------------------------------------
    # GES 结构学习 (使用 BDeu)
    # ------------------------------------------------------------------
    def learn_structure_ges(self, equivalent_sample_size=10.0):
        print("开始结构学习（GES算法，BDeu评分）...")
        score = BDeu(self.data, equivalent_sample_size=equivalent_sample_size)
        ges = GES(self.data)
        best_model = ges.estimate(scoring_method=score)
        print(f"✓ GES完成, 边数: {len(best_model.edges())}")
        return best_model

    # ------------------------------------------------------------------
    # MMHC 结构学习 (使用 BDeu)
    # ------------------------------------------------------------------
    def learn_structure_mmhc(self, significance_level=0.05,
                             equivalent_sample_size=10.0):
        print("开始结构学习（MMHC算法，BDeu评分）...")
        score = BDeu(self.data, equivalent_sample_size=equivalent_sample_size)
        mmhc = MmhcEstimator(self.data)
        best_model = mmhc.estimate(
            scoring_method=score,
            significance_level=significance_level
        )
        print(f"✓ MMHC完成, 边数: {len(best_model.edges())}")
        return best_model

    # ------------------------------------------------------------------
    # PC 结构学习
    # ------------------------------------------------------------------
    def learn_structure_pc_algorithm(self,
                                     significance_level=0.05,
                                     variant='stable',
                                     ci_test='chi_square'):
        print("开始结构学习（PC算法）...")
        pc = PC(self.data)
        best_model = pc.estimate(
            significance_level=significance_level,
            variant=variant,
            ci_test=ci_test
        )
        print(f"✓ PC完成, 边数: {len(best_model.edges())}")
        return best_model

    # ------------------------------------------------------------------
    # 保存结构到 JSON
    # ------------------------------------------------------------------
    def _save_structure_to_json(self, learned_bn, output_dir,
                                network_name="", learning_method="unknown"):
        node_info = {node: [str(v) for v in sorted(self.data[node].dropna().unique())]
                     for node in learned_bn.nodes()}
        json_structure = {
            "metadata": {
                "network_name": network_name,
                "description": "从数据中学习得到的贝叶斯网络",
                "creation_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "learning_method": learning_method,
                "data_shape": list(self.data.shape)
            },
            "individual_variables": node_info,
            "edges": {
                "single_edges": [{"from": e[0], "to": e[1]} for e in learned_bn.edges()]
            }
        }
        file_pre = os.path.basename(output_dir.rstrip('/'))
        json_path = os.path.join(output_dir, f"{file_pre}_learned_edges.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_structure, f, ensure_ascii=False, indent=2)
        return json_path

    # ------------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------------
    def learn_complete_model(self, structure_score_method, save_dir,
                             expert_knowledge_=None, beta=500,
                             equivalent_sample_size=10.0):
        method_upper = structure_score_method.upper()
        if method_upper == "PC":
            structure = self.learn_structure_pc_algorithm()
        elif method_upper == "GES":
            structure = self.learn_structure_ges(
                equivalent_sample_size=equivalent_sample_size)
        elif method_upper == "MMHC":
            structure = self.learn_structure_mmhc(
                equivalent_sample_size=equivalent_sample_size)
        else:
            strategy = structure_score_method.lower()
            structure = self.learn_structure_hill_climbing(
                strategy=strategy,
                equivalent_sample_size=equivalent_sample_size,
                ek=expert_knowledge_,
                beta=beta
            )
        self._save_structure_to_json(structure, save_dir,
                                     learning_method=structure_score_method)


# =========================================================================
# 数据路径扫描
# =========================================================================
def get_data_json(root_dir: str) -> dict:
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


def process_save_path(orig_path, tag):
    """根据样本数据目录生成用于保存估计结果的目录"""
    dir_path = os.path.dirname(orig_path)
    dir_path = dir_path.replace("data", f"results/{tag}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


# =========================================================================
# 加载专家知识（支持 alpha 动态重计算）
# =========================================================================
def load_ek_json_data(hier_blacklist_path_: str, mono_blacklist_path_: str,
                      alpha_override: Optional[float] = None) -> Dict:
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


# =========================================================================
# 主程序：遍历 ESS、alpha、beta
# =========================================================================
if __name__ == '__main__':
    base_path = "../experiments/generated_bns/"

    # 所有策略分类（与具体 ESS 无关）
    static_strategies = [
        # "PC", "GES", "MMHC", "HC_BDeu", "hierarchy"
        # "PC", "GES", "HC_BDeu", "hierarchy"
    ]
    alpha_sensitive_strategies = [
        # "monotonic_before", "monotonic_after", "monotonic_score",
        # "hierarchy_monotonic_before", "hierarchy_monotonic_after",
        "hierarchy_monotonic_score"
    ]

    # 超参数网格
    # ess_values = [1.0, 10.0]  # BDeu 等效样本大小
    ess_values = [1.0]
    # alpha_values = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    alpha_values = [0.02, 0.05]
    beta_values = [100, 200, 500, 1000]

    data_json_dict = get_data_json(base_path)

    for bn_name, bn_info in data_json_dict.items():
        for sample_path, mono_blacklist_path in bn_info["sample_mono"]:
            data = pd.read_csv(sample_path, sep=',')

            # ----- 静态策略 (对每个 ESS 都跑一次) -----
            for func in static_strategies:
                for ess_val in ess_values:
                    # 对于 PC, GES, MMHC，ESS 通过 equivalent_sample_size 控制；PC 忽略 ESS
                    tag = f"{func}_ess{int(ess_val)}" if func != "PC" else func
                    print("*" * 24, f"{sample_path} strategy={func} ESS={ess_val}", "*" * 24)
                    save_dir = process_save_path(sample_path, tag)
                    # PC 等不需要 ek，但为了统一调用，仍然加载 ek（仅 hierarchy 需要层级黑名单）
                    ek = load_ek_json_data(bn_info["hier_blacklist"], mono_blacklist_path)
                    learner = BayesianNetworkLearner(data)
                    learner.learn_complete_model(
                        structure_score_method=func,
                        save_dir=save_dir,
                        expert_knowledge_=ek if func in ("hierarchy",) else None,
                        beta=500,
                        equivalent_sample_size=ess_val
                    )

            # ----- alpha 敏感策略 (遍历 ESS、alpha、beta) -----
            for func in alpha_sensitive_strategies:
                is_score_based = 'score' in func
                for ess_val in ess_values:
                    for alpha_val in alpha_values:
                        alpha_tag = str(alpha_val).replace('.', 'd')
                        betas = beta_values if is_score_based else [500]
                        for beta_val in betas:
                            if is_score_based:
                                tag = f"{func}_ess{int(ess_val)}_alpha{alpha_tag}_beta{beta_val}"
                            else:
                                tag = f"{func}_ess{int(ess_val)}_alpha{alpha_tag}"
                            print("*" * 24, f"{sample_path} {tag}", "*" * 24)
                            save_dir = process_save_path(sample_path, tag)
                            ek = load_ek_json_data(
                                bn_info["hier_blacklist"],
                                mono_blacklist_path,
                                alpha_override=alpha_val
                            )
                            learner = BayesianNetworkLearner(data)
                            learner.learn_complete_model(
                                structure_score_method=func,
                                save_dir=save_dir,
                                expert_knowledge_=ek,
                                beta=beta_val,
                                equivalent_sample_size=ess_val
                            )

    print("所有结构学习任务完成。")