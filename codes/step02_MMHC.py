#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_mmhc_only.py
仅执行 MMHC 算法，遍历所有 BN 实例、所有样本量。
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from pgmpy.estimators import MmhcEstimator

# 屏蔽pgmpy冗余日志
import logging
logging.getLogger("pgmpy").setLevel(logging.WARNING)


class BNLearnerMMHC:
    """简化学习器：仅封装 MMHC 结构学习方法"""
    def __init__(self, data):
        self.data = data
        self.variable_names = list(data.columns)

    def run_mmhc(self, significance_level=0.05, score='bdeu'):
        """
        执行 MMHC 结构学习
        参数:
            significance_level: 独立性检验显著水平
            score: 评分函数，默认'bdeu'
        返回:
            learned_model (DAG)
        """
        print(f"  MMHC 参数: sig={significance_level}, score={score}")
        t0 = time.time()
        mmhc = MmhcEstimator(self.data)
        model = mmhc.estimate(
            scoring_method=score,
            significance_level=significance_level
        )
        t1 = time.time()
        print(f"  ✓ MMHC 完成，耗时 {t1 - t0:.2f} 秒，边数 {len(model.edges())}")
        return model

    def save_structure(self, learned_bn, output_dir, learning_method="MMHC"):
        """保存学到的结构为 JSON 文件"""
        node_info = {node: [str(v) for v in sorted(self.data[node].dropna().unique())]
                     for node in learned_bn.nodes()}
        json_structure = {
            "metadata": {
                "network_name": "",
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


def get_data_json(root_dir: str) -> dict:
    """扫描 BN 实例目录，返回 {bn_name: {sample_mono: [[csv, mono_json], ...], hier_blacklist: path}}"""
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

        # 过滤掉样本量 >= 1000 的目录
        s_dirs = [d for d in s_dirs if int(re.search(r'\d+', d.name).group()) < 1000]

        for s_dir in s_dirs:
            csv_files = list(s_dir.glob("*.csv"))
            if not csv_files:
                continue
            mono_files = list(s_dir.glob("*_mono_blacklist_alpha*.json"))
            # MMHC 不需要单调性文件，但保留信息以兼容后续处理
            sample_mono.append([f"./{csv_files[0].as_posix()}",
                                f"./{mono_files[0].as_posix()}" if mono_files else None])
        hier_file = bn_dir / "ground_truth" / "topo_hier_blacklist.json"
        hier_path = f"./{hier_file.as_posix()}" if hier_file.is_file() else None
        result[bn_name] = {
            "sample_mono": sample_mono,
            "hier_blacklist": hier_path
        }
    return result


def process_save_path(orig_csv_path, strategy_name):
    """将原数据路径转换为结果保存路径：.../data/S100/ -> .../results/MMHC/S100/"""
    dir_path = os.path.dirname(orig_csv_path)
    dir_path = dir_path.replace("data", f"results/{strategy_name}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


if __name__ == "__main__":
    BASE_DIR = "../experiments/generated_bns/"
    STRATEGY = "MMHC"

    # 参数设置（可根据需要调整）
    SIG_LEVEL = 0.05
    SCORE = "bdeu"

    data_dict = get_data_json(BASE_DIR)
    total_jobs = sum(len(v["sample_mono"]) for v in data_dict.values())
    print(f"共发现 {len(data_dict)} 个 BN 实例，总计 {total_jobs} 个样本配置")

    for bn_name, bn_info in data_dict.items():
        for csv_path, mono_path in bn_info["sample_mono"]:
            print(f"\n处理 {bn_name} -> {csv_path}")
            print(os.path.abspath(csv_path))
            if not os.path.exists(csv_path):
                print("  ⚠ 数据文件不存在，跳过")
                continue

            data = pd.read_csv(csv_path, sep=',')
            learner = BNLearnerMMHC(data)
            model = learner.run_mmhc(
                significance_level=SIG_LEVEL,
                score=SCORE
            )

            save_dir = process_save_path(csv_path, STRATEGY)
            learner.save_structure(model, save_dir, learning_method=STRATEGY)
            print(f"  结构已保存到 {save_dir}")

    print("\n所有 MMHC 任务完成。")