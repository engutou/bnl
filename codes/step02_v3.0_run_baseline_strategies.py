#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_baseline_strategies.py
独立运行四种基线策略：PC、GES、HC_BDeu、hierarchy。
不涉及 alpha/beta，只涉及 ESS（GES、HC_BDeu、hierarchy 需要区分 ESS=1 和 ESS=10）。
结果保存到 results/ 目录下，可直接被 step03_evaluate_models.py 读取。
"""

import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
from pgmpy.estimators import BDeu, ExpertKnowledge, HillClimbSearch, GES, PC
from pgmpy.base import DAG

logging.getLogger("pgmpy").setLevel(logging.WARNING)


# =========================================================================
# 简化版学习器：只封装基线策略
# =========================================================================
class BaselineLearner:
    def __init__(self, data):
        self.data = data
        self.variable_names = list(data.columns)
        print(f"数据形状: {self.data.shape}, 变量 {len(self.variable_names)} 个")

    def learn_hill_climbing(self, ess=10.0, forbidden_edges=None):
        """爬山法 + BDeu 评分，可选黑名单"""
        scorer = BDeu(self.data, equivalent_sample_size=ess)
        ek = ExpertKnowledge()
        if forbidden_edges:
            ek.forbidden_edges = forbidden_edges
        else:
            ek = None

        hc = HillClimbSearch(self.data)
        model = hc.estimate(scoring_method=scorer, expert_knowledge=ek,
                            show_progress=False)
        return model

    def learn_ges(self, ess=10.0):
        """GES + BDeu 评分"""
        scorer = BDeu(self.data, equivalent_sample_size=ess)
        ges = GES(self.data)
        model = ges.estimate(scoring_method=scorer)
        return model

    def learn_pc(self, significance_level=0.05):
        """PC-stable 算法"""
        pc = PC(self.data)
        model = pc.estimate(significance_level=significance_level,
                            variant='stable', ci_test='chi_square')
        return model

    def save_structure(self, model, output_dir, strategy_tag):
        """保存学到的结构为 JSON 文件"""
        node_info = {node: [str(v) for v in sorted(self.data[node].dropna().unique())]
                     for node in model.nodes()}
        json_structure = {
            "metadata": {
                "learning_method": strategy_tag,
                "data_shape": list(self.data.shape)
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
# 工具函数（与 step02 保持一致）
# =========================================================================
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


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
            # 基线策略不需要单调性 JSON，但保留路径以兼容后续评估脚本
            mono_files = list(s_dir.glob("*_mono_blacklist_alpha*.json"))
            sample_mono.append([f"./{csv_files[0].as_posix()}",
                                f"./{mono_files[0].as_posix()}" if mono_files else None])
        # 层级黑名单（hierarchy 需要）
        hier_file = bn_dir / "ground_truth" / "topo_hier_blacklist.json"
        if not hier_file.is_file():
            # 尝试备用文件名
            hier_file = bn_dir / "ground_truth" / "dag_hier_blacklist.json"
        if hier_file.is_file():
            result[bn_name] = {
                "sample_mono": sample_mono,
                "hier_blacklist": f"./{hier_file.as_posix()}"
            }
    return result


def process_save_path(orig_path, tag):
    """根据样本路径和策略标签生成结果保存目录。"""
    dir_path = os.path.dirname(orig_path)
    dir_path = dir_path.replace("data", f"results/{tag}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def result_exists(sample_csv_path, tag):
    """检查对应结果目录下是否已有 learned_edges.json 文件。"""
    res_dir = process_save_path(sample_csv_path, tag)
    json_files = list(Path(res_dir).glob("*_learned_edges.json"))
    return len(json_files) > 0


# =========================================================================
# 主程序
# =========================================================================
if __name__ == '__main__':
    base_path = "../experiments_N12_E16/generated_bns/"

    # 四种基线策略及其参数
    # 格式: (策略标识, 策略类型, 是否需要 ESS)
    # 策略类型: 'pc', 'ges', 'hc', 'hierarchy'
    baseline_strategies = [
        ('PC', 'pc', False),
        ('GES', 'ges', True),
        ('HC_BDeu', 'hc', True),
        ('hierarchy', 'hierarchy', True),
    ]

    ESS_VALUES = [1.0, 10.0]

    data_json_dict = get_data_json(base_path)
    total_jobs = 0
    skipped_jobs = 0

    for bn_name, bn_info in data_json_dict.items():
        for sample_csv_path, mono_json_path in bn_info["sample_mono"]:
            # 加载层级黑名单（hierarchy 需要）
            hier_blacklist = None
            if bn_info.get("hier_blacklist"):
                try:
                    hier_data = load_json(bn_info["hier_blacklist"])
                    hier_blacklist = [(h["from"], h["to"]) for h in hier_data.get("forbidden_edges_by_hier", [])]
                except Exception as e:
                    print(f"  警告: 无法加载层级黑名单 {bn_info['hier_blacklist']}: {e}")

            data = pd.read_csv(sample_csv_path, sep=',')

            for strategy_name, strategy_type, needs_ess in baseline_strategies:
                if needs_ess:
                    for ess_val in ESS_VALUES:
                        tag = f"{strategy_name}_ess{int(ess_val)}"

                        # 跳过已存在的
                        if result_exists(sample_csv_path, tag):
                            skipped_jobs += 1
                            continue

                        print("*" * 24, f"{sample_csv_path} {tag}", "*" * 24)

                        learner = BaselineLearner(data)

                        if strategy_type == 'ges':
                            model = learner.learn_ges(ess=ess_val)
                        elif strategy_type == 'hc':
                            model = learner.learn_hill_climbing(ess=ess_val)
                        elif strategy_type == 'hierarchy':
                            model = learner.learn_hill_climbing(ess=ess_val, forbidden_edges=hier_blacklist)
                        else:
                            continue  # 不应该到这里

                        save_dir = process_save_path(sample_csv_path, tag)
                        learner.save_structure(model, save_dir, tag)
                        total_jobs += 1
                        print(f"  ✓ 边数 {len(model.edges())} 保存至 {save_dir}")

                else:
                    # PC 不受 ESS 影响，只有一个版本
                    tag = "PC"

                    if result_exists(sample_csv_path, tag):
                        skipped_jobs += 1
                        continue

                    print("*" * 24, f"{sample_csv_path} {tag}", "*" * 24)

                    learner = BaselineLearner(data)
                    model = learner.learn_pc(significance_level=0.05)

                    save_dir = process_save_path(sample_csv_path, tag)
                    learner.save_structure(model, save_dir, tag)
                    total_jobs += 1
                    print(f"  ✓ 边数 {len(model.edges())} 保存至 {save_dir}")

    print(f"\n所有基线策略任务完成！")
    print(f"  新运行: {total_jobs} 个任务")
    print(f"  跳过已存在: {skipped_jobs} 个任务")