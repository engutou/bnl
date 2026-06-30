import json
import os
from collections import deque
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


def find_json_files(root_dir):
    """
    递归获取 root_dir 下所有 .json 文件路径
    """
    json_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('dag.json'):
                full_path = os.path.join(dirpath, file).replace("\\", "/")
                json_files.append(full_path)
    return json_files


# 示例调用
if __name__ == "__main__":
    base_path = "../generated_bns/"
    json_path_list = find_json_files(base_path)
    for json_path in json_path_list:
        print("*" * 30, "根据 dag 生成节点层级数据 dag_features", "*" * 30)
        node2levels, level2nodes, parent_map, child_map = dag_analysis(json_path)

        # 控制台打印验证（可选）
        print("\n=== 节点→层级（排序后）===")
        for node, level in sorted(node2levels.items(), key=lambda x: int(x[0][1:])):
            print(f"节点 {node}: 层级 {level}")

        print("\n=== 层级→节点列表（排序后）===")
        for level, nodes in sorted(level2nodes.items()):
            print(f"层级 {level}: {nodes}")

        print("\n=== 父节点映射（排序后，示例V3）===")
        print(f"V3的父节点: {parent_map['V3']}")

        print("\n=== 子节点映射（排序后，示例V1）===")
        print(f"V1的子节点: {child_map['V1']}")
