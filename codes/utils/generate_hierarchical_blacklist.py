from collections import deque
from itertools import product
from typing import List, Tuple, Dict
from warnings import warn

import networkx as nx


def is_dag_weakly_connected(all_nodes, edges):
    """用networkx校验DAG是否为弱连通图"""
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from([(edge['from'], edge['to']) for edge in edges])
    if len(G.nodes) == 0:
        return True
    return nx.is_weakly_connected(G)


def dag_analysis(json_file_path):
    """
    从JSON文件读取DAG，计算节点层级/父子映射，并将排序后的特征输出到JSON文件
    返回：node2levels, level2nodes, parent_map, child_map
    """
    # 1. 读取并解析JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dag_data = json.load(f)

    # 2. 提取所有节点和边信息
    all_nodes = list(dag_data['individual_variables'].keys())
    edges = dag_data['edges']['single_edges']

    # 3. DAG连通性校验
    if not is_dag_weakly_connected(all_nodes, edges):
        warn("警告：当前DAG不是连通图！请检查拓扑结构。")

    # 4. 构建入度/父节点/子节点字典
    in_degree = {node: 0 for node in all_nodes}
    parent_map = {node: [] for node in all_nodes}  # 每个节点的父节点列表
    child_map = {node: [] for node in all_nodes}  # 每个节点的子节点列表

    for edge in edges:
        from_node = edge['from']
        to_node = edge['to']
        in_degree[to_node] += 1
        parent_map[to_node].append(from_node)
        child_map[from_node].append(to_node)

    # 5. 拓扑排序计算节点层级
    node_levels = {}
    for node in all_nodes:
        node_levels[node] = 1 if in_degree[node] == 0 else 0

    queue = deque([node for node in all_nodes if in_degree[node] == 0])
    while queue:
        current_node = queue.popleft()
        children = [edge['to'] for edge in edges if edge['from'] == current_node]
        for child in children:
            parent_levels = [node_levels[p] for p in parent_map[child]]
            node_levels[child] = max(parent_levels) + 1
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # 6. 转换为「层级→节点列表」字典
    level_to_nodes = {}
    for node, level in node_levels.items():
        if level not in level_to_nodes:
            level_to_nodes[level] = []
        level_to_nodes[level].append(node)

    # ========== 核心：排序处理 ==========
    # 6.1 对parent_map/child_map的节点列表按名称排序（V1→V2→V3...）
    for node in parent_map:
        parent_map[node].sort(key=lambda x: int(x[1:]))  # 父节点列表排序
    for node in child_map:
        child_map[node].sort(key=lambda x: int(x[1:]))  # 子节点列表排序

    # 6.2 对node_levels按节点名称排序（转为有序字典）
    sorted_node2levels = dict(sorted(node_levels.items(), key=lambda x: int(x[0][1:])))

    # 6.3 对level_to_nodes按层级升序，且每个层级的节点列表按名称排序
    sorted_level2nodes = {}
    for level in sorted(level_to_nodes.keys()):
        sorted_nodes = sorted(level_to_nodes[level], key=lambda x: int(x[1:]))
        sorted_level2nodes[level] = sorted_nodes

    # 6.4 对parent_map/child_map按节点名称排序（转为有序字典）
    sorted_parent_map = dict(sorted(parent_map.items(), key=lambda x: int(x[0][1:])))
    sorted_child_map = dict(sorted(child_map.items(), key=lambda x: int(x[0][1:])))

    # ========== 写入JSON文件 ==========
    # 生成输出文件名：xx.json → xx_features.json
    dir_name = os.path.dirname(json_file_path)
    file_name = os.path.basename(json_file_path)
    new_file_name = file_name.replace('.json', '_features.json')
    output_path = os.path.join(dir_name, new_file_name)

    # 构建输出数据
    output_data = {
        "node_to_level": sorted_node2levels,  # 节点→层级（按节点名排序）
        "level_to_nodes": sorted_level2nodes,  # 层级→节点列表（按层级+节点名排序）
        "parent_map": sorted_parent_map,  # 节点→父节点列表（按节点名+父节点名排序）
        "child_map": sorted_child_map  # 节点→子节点列表（按节点名+子节点名排序）
    }

    # 写入JSON（保证中文/特殊字符正常，缩进4行提升可读性）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"特征文件已保存至：{output_path}")
    return node_levels, level_to_nodes, parent_map, child_map


def map_levels_balanced_three_layers(node_levels: dict) -> (dict, dict):
    """
    将拓扑层级映射为 1,2,3 三层，保证：
    - 原始层级为 1 的节点必定在 L1
    - 原始层级为最大值的节点必定在 L3
    - 三层的节点数量尽可能均衡
    """
    # 按层级排序的节点列表
    sorted_items = sorted(node_levels.items(), key=lambda x: x[1])
    nodes = [item[0] for item in sorted_items]
    levels = [item[1] for item in sorted_items]

    n = len(nodes)
    if n < 3:
        # 节点太少，直接按层级划分
        layer_mapping = {}
        for node, lvl in node_levels.items():
            if lvl == 1:
                layer_mapping[node] = 1
            elif lvl == max(levels):
                layer_mapping[node] = 3
            else:
                layer_mapping[node] = 2
        return layer_mapping

    max_lvl = max(levels)
    if max_lvl == 1:
        return {node: 1 for node in nodes}

    # 目标每层节点数
    target = n // 3

    # 寻找分割点1：使 L1 节点数接近 target，但不能切割同一层级的节点
    # 我们沿着排序后的节点累计，当累计数首次超过 target 且下一个节点的层级与当前不同时，确定分割边界
    cum = 0
    split1_idx = 0
    for i, lvl in enumerate(levels):
        cum += 1
        if cum >= target:
            # 确保不在同一层级内切割：检查下一个节点的层级是否与当前相同
            if i == n - 1 or levels[i + 1] != lvl:
                split1_idx = i + 1
                break
    # 如果没找到合适分割点，就取 target 位置
    if split1_idx == 0:
        split1_idx = min(target, n - 2)
    # 保证 L1 至少包含所有层级 1 的节点
    last_lvl1_idx = max(i for i, lvl in enumerate(levels) if lvl == 1)
    split1_idx = max(split1_idx, last_lvl1_idx + 1)

    # 寻找分割点2：使 L3 节点数接近 target，从后往前
    cum = 0
    split2_idx = n
    for i in range(n - 1, -1, -1):
        cum += 1
        if cum >= target:
            if i == 0 or levels[i - 1] != levels[i]:
                split2_idx = i
                break
    if split2_idx == n:
        split2_idx = max(n - target, split1_idx + 1)
    # 保证 L3 至少包含所有最大层级的节点
    first_max_idx = min(i for i, lvl in enumerate(levels) if lvl == max_lvl)
    split2_idx = min(split2_idx, first_max_idx)

    # 防止交叉
    if split1_idx >= split2_idx:
        # 退化为均分
        split1_idx = n // 3
        split2_idx = 2 * n // 3

    # 根据索引分配层级
    layer_mapping = {}
    for i, node in enumerate(nodes):
        if i < split1_idx:
            layer_mapping[node] = 1
        elif i < split2_idx:
            layer_mapping[node] = 2
        else:
            layer_mapping[node] = 3

    # 根据 mapped_levels 构建新的层级分组
    level2nodes = {1: [], 2: [], 3: []}
    for node, lvl in layer_mapping.items():
        level2nodes[lvl].append(node)
    for lvl in level2nodes:
        level2nodes[lvl].sort(key=lambda x: int(x[1:]))

    return layer_mapping, level2nodes


def generate_hierarchical_blacklist_from_level2nodes(
        level2nodes: Dict[int, List[str]]
) -> List[Tuple[str, str]]:
    """
    根据三层节点分组（level2nodes）生成层级禁止边列表。

    禁止规则：父节点所在层 > 子节点所在层。
    即：L2→L1, L3→L1, L3→L2 禁止。

    参数
    ----------
    level2nodes : Dict[int, List[str]]
        键为层号（1,2,3），值为该层包含的节点名列表。

    返回
    -------
    List[Tuple[str, str]]
        禁止边列表，每个元素为 (parent, child)。
    """
    L1 = set(level2nodes.get(1, []))
    L2 = set(level2nodes.get(2, []))
    L3 = set(level2nodes.get(3, []))

    forbidden = []
    # L2 → L1
    forbidden.extend(product(L2, L1))
    # L3 → L1
    forbidden.extend(product(L3, L1))
    # L3 → L2
    forbidden.extend(product(L3, L2))

    return forbidden


import os
import json
from datetime import datetime


def generate_hierarchical_blacklist(topo_file_path: str):
    """
    根据拓扑文件生成层级约束禁止边列表，并保存为 JSON 文件。

    参数
    ----------
    topo_file_path : str
        原始拓扑 JSON 文件路径（如 "dag.json"）。

    返回
    -------
    str
        生成的禁止边 JSON 文件路径。
    """
    # 1. 分析 DAG，获取原始层级
    node2levels, level2nodes_orig, parent_map, child_map = dag_analysis(topo_file_path)

    # 2. 压缩为均衡三层
    layer_mapping, new_level2nodes = map_levels_balanced_three_layers(node2levels)

    # 3. 生成层级禁止边列表
    forbidden_edges = generate_hierarchical_blacklist_from_level2nodes(new_level2nodes)

    # 4. 构造输出数据结构
    # 将禁止边转换为与原始 JSON 兼容的格式 [{"from": u, "to": v}, ...]
    forbidden_edges_formatted = [{"from": u, "to": v} for u, v in forbidden_edges]

    output_data = {
        "metadata": {
            "description": "Hierarchical forbidden edges based on balanced three-layer compression",
            "source_file": os.path.basename(topo_file_path),
            "generated_at": datetime.now().isoformat(),
            "total_nodes": len(node2levels),
            "layer_distribution": {
                "L1": len(new_level2nodes.get(1, [])),
                "L2": len(new_level2nodes.get(2, [])),
                "L3": len(new_level2nodes.get(3, []))
            },
            "total_forbidden_edges": len(forbidden_edges)
        },
        "layers": {
            "L1": sorted(new_level2nodes.get(1, []), key=lambda x: int(x[1:])),
            "L2": sorted(new_level2nodes.get(2, []), key=lambda x: int(x[1:])),
            "L3": sorted(new_level2nodes.get(3, []), key=lambda x: int(x[1:]))
        },
        "forbidden_edges_by_hier": forbidden_edges_formatted
    }

    # 5. 确定输出路径
    dir_name = os.path.dirname(topo_file_path)
    base_name = os.path.splitext(os.path.basename(topo_file_path))[0]
    output_path = os.path.join(dir_name, f"topo_hier_blacklist.json")

    # 6. 写入 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"层级禁止边文件已保存至: {output_path}")
    print(f"  - 节点总数: {output_data['metadata']['total_nodes']}")
    print(f"  - 层级分布: L1={output_data['metadata']['layer_distribution']['L1']}, "
          f"L2={output_data['metadata']['layer_distribution']['L2']}, "
          f"L3={output_data['metadata']['layer_distribution']['L3']}")
    print(f"  - 禁止边总数: {len(forbidden_edges)}")

    return output_path, forbidden_edges_formatted


# 示例调用
if __name__ == "__main__":
    json_path = "topo.json"  # 输入文件路径
    generate_hierarchical_blacklist(json_path)
    # node2levels, level2nodes, parent_map, child_map = dag_analysis(json_path)
    #
    # mapped_levels, new_level2nodes = map_levels_balanced_three_layers(node2levels)
    #
    # forbidden_edges = generate_hierarchical_blacklist_from_level2nodes(new_level2nodes)
    # print(forbidden_edges)

    # # 打印原始层级分组
    # print("\n" + "=" * 60)
    # print("原始拓扑层级分组 (level2nodes)")
    # print("=" * 60)
    # for lvl in sorted(level2nodes.keys()):
    #     nodes = level2nodes[lvl]
    #     print(f"层级 {lvl} ({len(nodes)}个): {nodes}")
    #
    # # 打印压缩后的三层分组
    # print("\n" + "=" * 60)
    # print("压缩后三层分组 (L1/L2/L3)")
    # print("=" * 60)
    # for lvl in [1, 2, 3]:
    #     nodes = new_level2nodes[lvl]
    #     print(f"L{lvl} ({len(nodes)}个): {nodes}")
    #
    # # 可选：打印每个节点的层级变化
    # print("\n" + "=" * 60)
    # print("节点层级映射对照 (原始层级 → 新层级)")
    # print("=" * 60)
    # for node in sorted(node2levels.keys(), key=lambda x: int(x[1:])):
    #     print(f"{node}: {node2levels[node]} → {mapped_levels[node]}")

    # 控制台打印验证（可选）
    # print("\n=== 节点→层级（排序后）===")
    # for node, level in sorted(node2levels.items(), key=lambda x: int(x[0][1:])):
    #     print(f"节点 {node}: 层级 {level}")

    # print("\n=== 层级→节点列表（排序后）===")
    # for level, nodes in sorted(level2nodes.items()):
    #     print(f"层级 {level}: {nodes}")

    # print("\n=== 父节点映射（排序后，示例V3）===")
    # print(f"V3的父节点: {parent_map['V3']}")
    #
    # print("\n=== 子节点映射（排序后，示例V1）===")
    # print(f"V1的子节点: {child_map['V1']}")
