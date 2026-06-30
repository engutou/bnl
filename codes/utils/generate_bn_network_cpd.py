import json
import os
import random
from itertools import product

import numpy as np
import pandas as pd


def generate_bayesian_network(
        num_nodes,
        target_edges,
        prob_range,
        effect_strength_range,
        max_parents,
        random_seed,
        folder_name):
    """
    生成贝叶斯网络结构（DAG）和所有节点的CPT表。
    节点命名：V1, V2, ..., Vn
    参数：
        num_nodes: 节点数量（默认24）
        target_edges: 目标边数（默认70）
        prob_range: 条件概率P(node=1)的取值范围（min, max）
        effect_strength_range: 每个父节点的影响强度范围（正数）
        max_parents: 每个节点的最大父节点数
        random_seed: 随机种子
    输出：
        在文件夹 bn_N{num_nodes}_E{target_edges}_p{min}_{max} 中保存：
            topo.json: 网络结构
            {node}_cpd.csv: 每个节点的CPT表
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 生成节点名称：V1, V2, ..., Vnum_nodes
    nodes = [f"V{i}" for i in range(1, num_nodes + 1)]

    # 1. 生成DAG拓扑结构（边只能从小序号指向大序号）
    def generate_dag(nodes, target_edges, max_parents):
        # 节点按序号顺序排列，索引从1开始，序号为 i
        edges = []
        parent_counts = {node: 0 for node in nodes}
        for idx, node in enumerate(nodes):
            # 可选父节点：序号小于当前节点的所有节点
            possible_parents = nodes[:idx]
            if idx == 0:
                n_parents = 0
            elif idx == 1:
                n_parents = random.randint(1, min(2, len(possible_parents)))
            else:
                # 大多数节点2-4个父节点，调整权重
                n_parents = random.choices([2, 3, 4], weights=[0.2, 0.6, 0.2])[0]
                n_parents = min(n_parents, len(possible_parents), max_parents)
            if n_parents > 0:
                chosen = random.sample(possible_parents, n_parents)
                for p in chosen:
                    edges.append((p, node))
                    parent_counts[node] += 1

        # 调整边数至接近目标
        current = len(edges)
        print(f"初始生成边数: {current}")
        if current < target_edges:
            # 所有可能的有向边（序号小→大）
            all_possible = []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    all_possible.append((nodes[i], nodes[j]))
            candidates = [e for e in all_possible if e not in edges]
            random.shuffle(candidates)
            for frm, to in candidates:
                if len(edges) >= target_edges:
                    break
                if parent_counts[to] < max_parents:
                    edges.append((frm, to))
                    parent_counts[to] += 1

        elif current > target_edges:
            remove_cnt = current - target_edges

            # 计算每个节点的入度、出度、总度数
            in_degree = parent_counts.copy()  # 入度字典
            out_degree = {node: 0 for node in nodes}  # 出度字典
            for frm, _ in edges:
                out_degree[frm] += 1
            total_degree = {node: in_degree[node] + out_degree[node] for node in nodes}

            # 按目标节点的入度降序排序（与原逻辑一致，也可改用其他指标）
            edges_sorted = sorted(edges, key=lambda e: in_degree[e[1]], reverse=True)

            removed = 0
            for e in edges_sorted:
                if removed >= remove_cnt:
                    break
                frm, to = e
                # 检查删除这条边是否会导致 frm 或 to 变成孤立节点
                if total_degree[frm] == 1 or total_degree[to] == 1:
                    continue  # 跳过，保留这条边
                # 安全删除
                edges.remove(e)
                # 更新度数
                out_degree[frm] -= 1
                in_degree[to] -= 1
                total_degree[frm] = in_degree[frm] + out_degree[frm]
                total_degree[to] = in_degree[to] + out_degree[to]
                removed += 1

            if removed < remove_cnt:
                print(f"⚠ 为避免孤立节点，仅删除了 {removed} 条边，目标 {remove_cnt} 条。最终边数 = {len(edges)}")

        print(f"最终边数: {len(edges)}")
        return edges

    edges = generate_dag(nodes, target_edges, max_parents)
    topo = {
        "individual_variables": {node: ["0", "1"] for node in nodes},
        "edges": {
            "single_edges": [{"from": frm, "to": to} for frm, to in edges]
        },
        "metadata": {
            "贝叶斯网络名称": f"BN_{num_nodes}nodes_{len(edges)}edges",
            "网络描述": f"随机生成的有向无环图，节点数{num_nodes}，边数{len(edges)}，概率范围{prob_range}",
            "版本": "1.0",
            "创建时间": "2026-04-22",
            "节点数": f"{num_nodes}",
            "边数": f"{len(edges)}",
            "最大父节点数": f"{max_parents}",
            "根节点故障概率范围": f"{prob_range}",
            "节点相关性强度系数": f"{effect_strength_range}"
        }
    }

    # 创建输出文件夹
    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, "dag.json"), "w", encoding="utf-8") as f:
        json.dump(topo, f, indent=2, ensure_ascii=False)
    print(f"拓扑结构已保存到 {folder_name}/dag.json")

    # 构建父节点字典
    parents_dict = {node: [] for node in nodes}
    for frm, to in edges:
        parents_dict[to].append(frm)

    # 2. 生成CPT（保证父节点从0→1时，P(node=1)增加）
    min_prob, max_prob = prob_range
    min_effect, max_effect = effect_strength_range

    def generate_cpt(node, parents):
        parent_names = sorted(parents)  # 按字符串排序，V1, V2,... 自然顺序
        if not parent_names:
            # 根节点
            prob1 = round(random.uniform(min_prob, max_prob), 6)
            prob0 = 1 - prob1
            df = pd.DataFrame({
                "target_variable": [node],
                "parents": [""],
                "target_values": ["0,1"],
                f"{node}=0": [prob0],
                f"{node}=1": [prob1]
            })
            return df

        # 对于有父节点的情况，首先随机生成一个基准概率：所有父节点取0时，子节点取1的概率。
        base_prob = round(random.uniform(min_prob, max_prob), 6)
        # 为每个父节点分配一个正向影响强度，取值范围来自effect_strength_range（默认0.1~0.3）
        effects = {p: round(random.uniform(min_effect, max_effect), 6) for p in parent_names}

        rows = []
        """
        父节点是0，子节点是0的概率，大于父节点是1，子节点是0的概率；
        父节点是1，子节点是1的概率，大于父节点是0，子节点是1的概率
        """
        # 转换成v1（parent_1）的形式
        parent_cols = [f"{p}(parent_{idx})" for idx, p in enumerate(parent_names, start=1)]
        # 生成所有父节点取值的组合（笛卡尔积）
        dkej = product([0, 1], repeat=len(parent_names))
        for combo in dkej:
            # 计算基准概率对应的logit（对数几率）：ln(p/(1-p))。这是逻辑回归模型的截距项。
            logit = np.log(base_prob / (1 - base_prob))
            # 遍历当前组合的每个父节点及其取值val（0或1）。计算2*val-1：当val=0时结果为-1，当val=1时结果为+1。
            # 将该父节点的影响强度乘以-1或+1，累加到logit上。效果：若父节点取0，则减去该强度（降低子节点为1的对数几率）；
            # 若父节点取1，则加上该强度（升高对数几率）。这样就保证了父节点从0→1时，子节点取1的概率单调增加。
            for p, val in zip(parent_names, combo):
                # 只对父节点为1时候进行处理
                if val == 1:
                    logit += effects[p] * (2 * val - 1)  # val=0 → -effect, val=1 → +effect
            # 将累加后的logit通过sigmoid函数转换为概率值
            prob1 = 1 / (1 + np.exp(-logit))
            # 将计算出的概率裁剪到[0.05, 0.95]区间
            prob1 = round(np.clip(prob1, 0.05, 0.95), 6)

            prob0 = round(1 - prob1, 6)

            row = {
                "target_variable": node,
                "parents": ",".join(parent_names),
                "target_values": "0,1"
            }

            for p, val in zip(parent_cols, combo):
                row[p] = val
            row[f"{node}=0"] = prob0
            row[f"{node}=1"] = prob1
            rows.append(row)

        columns = ["target_variable", "parents", "target_values"] + parent_cols + [f"{node}=0", f"{node}=1"]
        return pd.DataFrame(rows, columns=columns)

    # 生成并保存每个节点的CPT
    c = 0
    for node in nodes:
        c += 1
        parents = parents_dict[node]
        df = generate_cpt(node, parents)
        filename = os.path.join(folder_name, f"{node}_cpd.csv")
        df.to_csv(filename, sep=";", index=False)
    print(f"所有文件已生成在文件夹: {folder_name}，保存了 {c} 个cpd.csv文件，")
