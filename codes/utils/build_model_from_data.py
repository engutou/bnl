# -*- coding:utf-8 -*-
# @Email     :jasonchujun@sina.com
# @Project   :DBN
# @FileName  :build_model_from_data.py
# @Time      :2025/12/24 17:17
# @Author    :Jason


"""
贝叶斯网络构建模块
提供从JSON文件加载网络结构、生成/读取CPD表格、构建完整贝叶斯网络的功能
"""

import json
import os
import re
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


class MyBayesianNetwork:
    """
    我的贝叶斯网络类 - 封装了pgmpy的贝叶斯网络功能

    主要功能:
    1. 从json文件中读取贝叶斯网络的结构信息
    2. 初始化DiscreteBayesianNetwork
    3. 根据贝叶斯网络的结构信息为每个节点生成csv格式的CPD文件
    4. 从CSV文件读取CPD并添加到模型
    5. 验证网络结构和CPD数据的正确性

    适用于工业控制系统(工控)场景的故障诊断和推理
    """

    def __init__(self):
        """
        初始化贝叶斯网络对象

        属性说明:
        - variables: 字典，存储变量名到其取值列表的映射，如 {'X_1': ['0','1','2']}
        - parents: 默认字典，存储每个变量的父节点列表，如 {'X_2': ['X_1']}
        - children: 默认字典，存储每个变量的子节点列表，如 {'X_1': ['X_2']}
        - model: pgmpy的离散贝叶斯网络模型对象
        - data_dir: 存储json文件和cpd文件的目录路径
        """
        self.variables = {}  # 变量名 -> 取值列表
        self.parents = defaultdict(list)  # 变量 -> 父节点列表
        self.children = defaultdict(list)  # 变量 -> 子节点列表
        self.model = None  # pgmpy的贝叶斯网络模型
        self.data_dir = None  # 数据文件所在目录

    def _set_model(self):
        """
        根据贝叶斯网络的结构初始化pgmpy的DiscreteBayesianNetwork模型

        说明:
        - 将网络中的边转换为pgmpy所需的格式
        - 创建空的贝叶斯网络模型，CPD稍后添加
        """
        # 将父节点关系转换为边列表，格式为[(from_var, to_var), ...]
        bn_edges = []
        for to_var, from_vars in self.parents.items():
            for from_var in from_vars:
                bn_edges.append((from_var, to_var))

        # 使用pgmpy创建离散贝叶斯网络模型
        self.model = DiscreteBayesianNetwork(bn_edges)

    def _add_variable(self, name, values):
        """
        添加变量到网络中
        'X_1': ['0','1','2']
        参数:
        - name: 变量名，字符串类型
        - values: 变量的可能取值列表，如 ['0','1','2','3']

        异常:
        - ValueError: 如果变量已存在或取值列表少于2个
        """
        # 检查变量是否已存在
        if name in self.variables:
            raise ValueError(f"变量 '{name}' 已存在")

        # 验证取值列表：必须是列表且至少有两个取值
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f"变量 '{name}' 必须至少有两个可能的取值")

        # 存储变量信息
        self.variables[name] = values

    def _add_edge(self, from_var, to_var):
        """
        在变量之间添加有向边（表示因果关系）

        参数:
        - from_var: 边的起始节点（父节点）
        - to_var: 边的结束节点（子节点）

        异常:
        - ValueError: 如果节点不存在

        说明: 添加边会同时更新父节点和子节点关系
        """
        # 验证两个变量是否都存在
        if from_var not in self.variables:
            raise ValueError(f"变量 '{from_var}' 不存在")
        if to_var not in self.variables:
            raise ValueError(f"变量 '{to_var}' 不存在")

        # 更新子节点关系：from_var -> to_var
        if to_var not in self.children[from_var]:
            self.children[from_var].append(to_var)

        # 更新父节点关系：from_var 是 to_var 的父节点
        if from_var not in self.parents[to_var]:
            self.parents[to_var].append(from_var)

    def _is_valid_network(self):
        """
        验证网络是否为有效的有向无环图(DAG)

        返回:
        - True: 网络是有效的DAG
        - False: 网络存在环（不是DAG）

        算法说明:
        使用深度优先搜索(DFS)检测图中是否有环
        维护两个集合：
        - visited: 已完全处理完成的节点
        - recursion_stack: 当前递归路径上的节点
        如果在递归过程中遇到已在recursion_stack中的节点，说明有环
        """
        visited = set()  # 已完成的节点
        recursion_stack = set()  # 当前递归路径上的节点

        def has_cycle(node):
            """
            递归检查从node开始的路径是否有环

            参数:
            - node: 当前检查的节点

            返回:
            - True: 发现环
            - False: 无环
            """
            if node in recursion_stack:
                # 当前节点已在递归栈中，说明存在环
                return True
            if node in visited:
                # 已处理过且确认无环的节点
                return False

            # 标记当前节点为正在处理
            visited.add(node)
            recursion_stack.add(node)

            # 递归检查所有子节点
            for child in self.children.get(node, []):
                if has_cycle(child):
                    return True

            # 当前节点处理完成，从递归栈移除
            recursion_stack.remove(node)
            return False

        # 对所有节点进行环检测
        for node in self.variables:
            if has_cycle(node):
                return False
        return True

    def _generate_valid_probabilities(self, target_card: int, parent_combination: tuple = None,
                                      parents: list = None) -> list:
        """
        生成一个变量的符合逻辑的概率分布（确保和为1，且贴合工控场景）

        注意: 这只是生成CSV格式的CPD文件模板时使用，实际贝叶斯网络推理时不用它产生的数据

        参数:
        - target_card: 目标变量的取值数量（基数）
        - parent_combination: 当前父节点取值组合，如('0','1')
        - parents: 父节点列表

        返回:
        - 归一化后的概率列表，长度等于target_card

        算法说明:
        1. 根节点（无父节点）：第一个取值（正常状态）概率为0.8-0.9，其余均匀分配
        2. 子节点：父节点异常等级越高，子节点正常状态概率越低
        3. 使用狄利克雷分布生成随机但归一化的概率
        """
        # 根节点处理（无父节点）
        if parent_combination is None:
            # 正常状态（第一个取值）概率 0.8-0.9，其余取值分配剩余概率
            base_probs = np.random.uniform(0.8, 0.9, size=1)  # 正常状态概率
            # 使用狄利克雷分布生成其他状态的概率（自动归一化）
            remaining_probs = np.random.dirichlet(np.ones(target_card - 1)) * (1 - base_probs[0])
            probs = np.concatenate([base_probs, remaining_probs])
        else:
            # 子节点处理：父节点状态越"异常"，子节点异常概率越高
            # 假设父节点取值为字符串类型的数字（如"0"=正常，"3"=完全失效）
            parent_abnormal_level = 0
            for p_val in parent_combination:
                try:
                    # 累加所有父节点的异常等级（取值越大越异常）
                    parent_abnormal_level += int(p_val)
                except (ValueError, TypeError):
                    # 非数字取值按1计算（默认轻度异常）
                    parent_abnormal_level += 1

            # 异常等级越高，子节点正常状态概率越低
            # 公式: normal_prob = max(0.1, 0.9 - parent_abnormal_level * 0.15)
            # 最低保留10%正常概率
            normal_prob = max(0.1, 0.9 - parent_abnormal_level * 0.15)
            base_probs = np.array([normal_prob])
            # 生成其他状态的概率
            remaining_probs = np.random.dirichlet(np.ones(target_card - 1)) * (1 - normal_prob)
            probs = np.concatenate([base_probs, remaining_probs])

        # 确保概率和严格等于1（解决浮点误差）
        probs = probs / probs.sum()
        probs = probs.round(4)

        # 二次检查，防止浮点误差导致概率和不为1
        total = probs.sum()
        if not np.isclose(total, 1.0):
            # 找到最接近0.5的概率值进行调整（减少对分布的影响）
            mid_idx = np.abs(probs - 0.5).argmin()
            probs[mid_idx] += (1.0 - total)

        return probs.round(4).tolist()

    def get_parents(self, variable_name):
        """
        获取指定变量的所有父节点

        参数:
        - variable_name: 变量名

        返回:
        - 父节点列表的副本

        异常:
        - ValueError: 如果变量不存在
        """
        if variable_name not in self.variables:
            raise ValueError(f"变量 '{variable_name}' 不存在")
        return self.parents[variable_name].copy()

    def get_children(self, variable_name):
        """
        获取指定变量的所有子节点

        参数:
        - variable_name: 变量名

        返回:
        - 子节点列表的副本

        异常:
        - ValueError: 如果变量不存在
        """
        if variable_name not in self.variables:
            raise ValueError(f"变量 '{variable_name}' 不存在")
        return self.children[variable_name].copy()

    def get_all_variables_name(self):
        """
        获取网络中所有变量的名称

        返回:
        - 变量名列表
        """
        return list(self.variables.keys())

    def get_variable_values(self, variable_name):
        """
        获取指定变量的可能取值列表

        参数:
        - variable_name: 变量名

        返回:
        - 变量的取值列表

        异常:
        - ValueError: 如果变量不存在
        """
        if variable_name not in self.variables:
            raise ValueError(f"变量 '{variable_name}' 不存在")
        return self.variables[variable_name]

    def generate_all_cpd_csv(self):
        """
        为网络中所有节点自动随机生成CPD CSV文件

        文件命名格式: var_name_cpd.csv
        文件格式: 分号分隔的CSV，包含表头和数据行

        说明:
        - 生成的文件是模板，需要用户根据实际情况修改概率值
        - 如果未指定data_dir，使用当前工作目录
        """
        # 检查网络是否已加载
        if not self.variables:
            raise RuntimeError("请先加载贝叶斯网络结构（如调用from_json_file方法）")

        # 设置数据目录
        if self.data_dir is None:
            self.data_dir = os.getcwd()  # 默认使用当前工作目录
            print(f"警告：未指定数据目录，将使用当前工作目录：{self.data_dir}")

        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)

        # 遍历所有变量，逐个生成CPD
        for var_name in self.get_all_variables_name():
            # 1. 获取目标变量的核心信息
            target_values = self.get_variable_values(var_name)  # 目标变量取值
            parents = self.get_parents(var_name)  # 父节点列表
            target_card = len(target_values)  # 目标变量状态个数

            # 2. 构建CSV表头字段
            # 基础字段：目标变量名、父节点列表、目标变量取值
            header = ["target_variable", "parents", "target_values"]

            # 父节点列字段（格式：父节点名(parent_序号)）
            if parents:
                parent_cols = [f"{p}(parent_{i + 1})" for i, p in enumerate(parents)]
                header.extend(parent_cols)
            else:
                # 无父节点时，使用占位列
                parent_cols = ["parent_placeholder"]
                header.extend(parent_cols)

            # 概率列字段（格式：var_name=取值）
            prob_cols = [f"{var_name}={val}" for val in target_values]
            header.extend(prob_cols)

            # 3. 生成CSV行数据
            rows = []

            # 生成父节点所有可能的取值组合（笛卡尔积）
            if parents:
                parent_value_lists = [self.get_variable_values(p) for p in parents]
                parent_combinations = list(product(*parent_value_lists))
            else:
                # 无父节点时，仅一行数据（先验分布）
                parent_combinations = [("None",)]

            # 为每个父节点组合生成一行CPD数据
            for combo in parent_combinations:
                # 初始化行数据，填充基础字段
                row = [
                    var_name,  # 目标变量名
                    ",".join(parents) if parents else "None",  # 父节点列表
                    ",".join(target_values)  # 目标变量取值
                ]

                # 填充父节点取值
                row.extend(combo)

                # 生成并填充概率
                probs = self._generate_valid_probabilities(
                    target_card=target_card,
                    parent_combination=combo if parents else None,
                    parents=parents
                )
                row.extend(probs)

                rows.append(row)

            # 4. 构建DataFrame并写入CSV
            df = pd.DataFrame(rows, columns=header)
            csv_filename = f"{var_name}_cpd.csv"
            csv_path = os.path.join(self.data_dir, csv_filename)

            # 保存为分号分隔的CSV文件
            df.to_csv(csv_path, sep=";", index=False, encoding="utf-8")
            print(f"成功生成CPD文件：{csv_path}")

    def read_cpd_from_csv(self, csv_filename):
        """
        从data_dir目录读取CPD文件，但不添加到模型（用于验证和调试）

        参数:
        - csv_filename: CPD文件名

        返回:
        - TabularCPD对象

        说明:
        - 与add_cpd_from_csv的区别：不要求CPD包含所有父节点
        - 父节点可以是网络结构定义中父节点的子集
        - 用于验证部分CPD数据
        """
        # 构建完整文件路径
        if self.data_dir is not None:
            csv_path = os.path.join(self.data_dir, csv_filename)
        else:
            csv_path = csv_filename  # 未指定目录时使用传入的路径

        try:
            # 1. 读取CSV文件，指定分号分隔
            df = pd.read_csv(csv_path, sep=';', comment='#')
            if df.empty:
                raise ValueError(f"CPD文件 {csv_path} 为空")

            # 2. 获取目标变量名
            target_var = df['target_variable'].iloc[0]
            if target_var not in self.variables:
                raise ValueError(f"目标变量 {target_var} 不在网络中")

            # 3. 获取目标变量取值
            target_values = [str(v).strip() for v in df['target_values'].iloc[0].split(',')]
            try:
                target_values_json = self.get_variable_values(target_var)
            except ValueError:
                raise ValueError(f"变量 {target_var} 未在JSON结构文件中定义")

            # 验证取值内容是否一致（不验证顺序）
            if sorted(target_values) != sorted(target_values_json):
                raise ValueError(
                    f"变量 {target_var} 的取值在CSV与JSON中不一致！\n"
                    f"CSV定义: {target_values}\n"
                    f"JSON定义: {target_values_json}"
                )

            # 4. 提取概率列
            prob_cols = [col.strip() for col in df.columns if f'{target_var}=' in col]

            # 从概率列名中提取取值（如从"X_3=0"中提取"0"）
            prob_col_values = [col.split('=')[1].strip() for col in prob_cols]

            # 验证概率列命名与目标取值一致（包括顺序）
            if prob_col_values != target_values:
                raise ValueError(
                    f"变量 {target_var} 的概率列命名与目标取值不匹配！\n"
                    f"概率列对应的取值: {prob_col_values}\n"
                    f"目标变量的取值: {target_values}"
                )

            # 5. 验证概率和为1
            prob_matrix = df[prob_cols].values  # 原始概率矩阵（未转置）
            row_sums = prob_matrix.sum(axis=1)  # 计算每行的概率和

            # 允许±1e-6的浮点误差
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                invalid_rows = [i for i, sum_val in enumerate(row_sums) if not np.isclose(sum_val, 1.0, atol=1e-6)]
                raise ValueError(
                    f"变量 {target_var} 的CPD概率和不为1！\n"
                    f"异常行索引: {invalid_rows}\n"
                    f"对应概率和: {[row_sums[i] for i in invalid_rows]}"
                )

            # 6. 提取父节点信息
            parents_raw = df['parents'].iloc[0]

            # 处理空父节点的情况
            if pd.isna(parents_raw) or str(parents_raw).strip().lower() in ['', 'none', 'nan']:
                parents = []  # 无父节点
            else:
                parents = [p.strip() for p in str(parents_raw).split(',')]

            # 7. 验证父节点存在
            for p in parents:
                if p not in self.variables:
                    raise ValueError(f"父节点 {p} 不在网络中")

            # 8. 验证父节点是网络结构中定义的父节点的子集（关键区别点）
            actual_parents = self.get_parents(target_var)
            if not set(parents).issubset(set(actual_parents)):
                raise ValueError(
                    f"CPD父节点和网络结构不符：{set(parents).difference(actual_parents)}未定义"
                )

            # 9. 提取父节点取值列
            parent_cols = [col for col in df.columns if '(parent_' in col]

            # 验证父节点列顺序与parents字段一致
            expected_parent_cols = [f"{p}(parent_{i + 1})" for i, p in enumerate(parents)]
            if parent_cols != expected_parent_cols:
                raise ValueError(
                    f"父节点列顺序与parents字段不符！\n"
                    f"预期列（按parents顺序）: {expected_parent_cols}\n"
                    f"实际列: {parent_cols}"
                )

            # 10. 验证父节点取值顺序符合标准笛卡尔积顺序
            parent_value_lists = [self.get_variable_values(p) for p in parents]
            standard_combinations = list(product(*parent_value_lists))

            # 解析CSV中的父节点组合
            csv_combinations = []
            for _, row in df.iterrows():
                combo = tuple(str(row[col]).strip() for col in parent_cols)
                csv_combinations.append(combo)

            # 比较标准顺序与实际顺序
            if standard_combinations != csv_combinations:
                raise ValueError(
                    f"父节点取值顺序与标准笛卡尔积的顺序不符！\n"
                    f"标准顺序: {standard_combinations}\n"
                    f"实际顺序: {csv_combinations}"
                )

            # 11. 准备创建TabularCPD所需的数据
            parent_cardinalities = [len(self.get_variable_values(p)) for p in parents]
            target_cardinality = len(target_values)

            # 转置概率矩阵，适配pgmpy的TabularCPD格式
            # 转置后：每行对应目标变量的一个取值，每列对应一组父节点取值组合
            prob_matrix = df[prob_cols].values.T

            # 12. 创建状态名称字典
            state_names = {target_var: target_values}
            for parent in parents:
                state_names[parent] = self.get_variable_values(parent)

            # 13. 创建TabularCPD对象
            cpd = TabularCPD(
                variable=target_var,
                variable_card=target_cardinality,
                values=prob_matrix,
                evidence=parents,
                evidence_card=parent_cardinalities,
                state_names=state_names
            )

            print(f"成功为变量 {target_var} 读取部分CPD（文件：{csv_path}）")
            return cpd

        except FileNotFoundError:
            raise FileNotFoundError(f"CPD文件 {csv_path} 不存在")
        except pd.errors.ParserError:
            raise ValueError(f"文件 {csv_path} 格式错误，请检查分号分隔是否正确")
        except IndexError:
            raise ValueError(f"文件 {csv_path} 缺少必要字段（如target_variable、parents等）")
        except Exception as e:
            raise RuntimeError(f"处理CPD时发生错误：{str(e)}")

    def add_cpd_from_csv(self, csv_filename):
        """
        从data_dir目录读取CPD文件并添加到模型

        参数:
        - csv_filename: CPD文件名

        说明:
        - 与read_part_cpd_from_csv的主要区别：要求CPD包含所有父节点
        - 父节点必须与网络结构完全一致
        """
        # 检查模型是否已初始化
        if self.model is None:
            raise RuntimeError("请先调用set_model()初始化模型")

        # 构建完整文件路径
        if self.data_dir is not None:
            csv_path = os.path.join(self.data_dir, csv_filename)
        else:
            csv_path = csv_filename

        try:
            # 1. 读取CSV文件
            df = pd.read_csv(csv_path, sep=';', comment='#')
            if df.empty:
                raise ValueError(f"CPD文件 {csv_path} 为空")

            # 2. 获取并验证目标变量
            target_var = df['target_variable'].iloc[0]
            if target_var not in self.variables:
                raise ValueError(f"目标变量 {target_var} 不在网络中")

            # 3. 获取并验证目标变量取值
            target_values = [str(v).strip() for v in df['target_values'].iloc[0].split(',')]
            try:
                target_values_json = self.get_variable_values(target_var)
            except ValueError:
                raise ValueError(f"变量 {target_var} 未在JSON结构文件中定义")

            if sorted(target_values) != sorted(target_values_json):
                raise ValueError(
                    f"变量 {target_var} 的取值在CSV与JSON中不一致！\n"
                    f"CSV定义: {target_values}\n"
                    f"JSON定义: {target_values_json}"
                )

            # 4. 提取并验证概率列
            prob_cols = [col.strip() for col in df.columns if f'{target_var}=' in col]
            prob_col_values = [col.split('=')[1].strip() for col in prob_cols]

            if prob_col_values != target_values:
                raise ValueError(
                    f"变量 {target_var} 的概率列命名与目标取值不匹配！\n"
                    f"概率列对应的取值: {prob_col_values}\n"
                    f"目标变量的取值: {target_values}"
                )

            # 5. 验证概率和为1
            prob_matrix = df[prob_cols].values
            row_sums = prob_matrix.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                invalid_rows = [i for i, sum_val in enumerate(row_sums) if not np.isclose(sum_val, 1.0, atol=1e-6)]
                raise ValueError(
                    f"变量 {target_var} 的CPD概率和不为1！\n"
                    f"异常行索引: {invalid_rows}\n"
                    f"对应概率和: {[row_sums[i] for i in invalid_rows]}"
                )

            # 6. 提取父节点信息
            parents_raw = df['parents'].iloc[0]
            if pd.isna(parents_raw) or str(parents_raw).strip().lower() in ['', 'none', 'nan']:
                parents = []
            else:
                parents = [p.strip() for p in str(parents_raw).split(',')]

            # 7. 验证父节点存在
            for p in parents:
                if p not in self.variables:
                    raise ValueError(f"父节点 {p} 不在网络中")

            # 8. 验证父节点关系与网络结构完全一致（关键区别点）
            actual_parents = self.get_parents(target_var)
            if sorted(parents) != sorted(actual_parents):
                raise ValueError(
                    f"CPD父节点与网络结构不符：定义 {parents}，实际 {actual_parents}"
                )

            # 9. 提取并验证父节点列
            parent_cols = [col for col in df.columns if '(parent_' in col]
            expected_parent_cols = [f"{p}(parent_{i + 1})" for i, p in enumerate(parents)]
            if parent_cols != expected_parent_cols:
                raise ValueError(
                    f"父节点列顺序与parents字段不符！\n"
                    f"预期列（按parents顺序）: {expected_parent_cols}\n"
                    f"实际列: {parent_cols}"
                )

            # 10. 验证父节点取值顺序
            parent_value_lists = [self.get_variable_values(p) for p in parents]
            standard_combinations = list(product(*parent_value_lists))

            csv_combinations = []
            for _, row in df.iterrows():
                combo = tuple(str(row[col]).strip() for col in parent_cols)
                csv_combinations.append(combo)

            if standard_combinations != csv_combinations:
                raise ValueError(
                    f"父节点取值顺序与标准笛卡尔积的顺序不符！\n"
                    f"标准顺序: {standard_combinations}\n"
                    f"实际顺序: {csv_combinations}"
                )

            # 11. 准备创建TabularCPD所需的数据
            parent_cardinalities = [len(self.get_variable_values(p)) for p in parents]
            target_cardinality = len(target_values)
            prob_matrix = df[prob_cols].values.T

            # 12. 创建状态名称字典
            state_names = {target_var: target_values}
            for parent in parents:
                state_names[parent] = self.get_variable_values(parent)

            # 13. 创建并添加TabularCPD对象到模型
            cpd = TabularCPD(
                variable=target_var,
                variable_card=target_cardinality,
                values=prob_matrix,
                evidence=parents,
                evidence_card=parent_cardinalities,
                state_names=state_names
            )

            self.model.add_cpds(cpd)
            # print(f"成功为变量 {target_var} 添加CPD（文件：{csv_path}）")

        except FileNotFoundError:
            raise FileNotFoundError(f"CPD文件 {csv_path} 不存在")
        except pd.errors.ParserError:
            raise ValueError(f"文件 {csv_path} 格式错误，请检查分号分隔是否正确")
        except IndexError:
            raise ValueError(f"文件 {csv_path} 缺少必要字段（如target_variable、parents等）")
        except Exception as e:
            raise RuntimeError(f"处理CPD时发生错误：{str(e)}")

    @classmethod
    def load_model_from_json_file(cls, filename):
        """
        从JSON文件加载贝叶斯网络结构（类方法）

        参数:
        - filename: JSON文件路径

        返回:
        - MyBayesianNetwork对象实例

        JSON文件格式说明:
        {
          "variable_groups": [  # 变量组定义
            {
              "name": "组名",
              "prefix": "变量前缀",
              "suffix": "变量后缀",
              "start": 起始编号,
              "end": 结束编号,
              "values": ["取值1", "取值2", ...]
            }
          ],
          "individual_variables": {  # 单独变量定义
            "变量名": ["取值1", "取值2", ...]
          },
          "edges": {  # 边定义
            "single_edges": [  # 单个边
              {"from": "父节点", "to": "子节点"}
            ],
            "group_edges": [  # 组间连接
              {
                "from_group": "源组名",
                "to_group": "目标组名",
                "connection_type": "one_to_one" 或 "all_to_all"
              }
            ]
          }
        }
        """
        try:
            # 获取JSON文件的绝对路径和所在目录
            file_path = os.path.abspath(filename)
            data_dir = os.path.dirname(file_path)

            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 创建网络实例并设置数据目录
            network = cls()
            network.data_dir = data_dir

            # 1. 处理变量组
            variable_groups = data.get('variable_groups', [])
            group_dict = {}  # 组名 -> 变量列表

            for group in variable_groups:
                group_name = group.get('name')
                prefix = group.get('prefix', '')
                suffix = group.get('suffix', '')
                start = group.get('start', 1)
                end = group.get('end', 1)
                values = group.get('values', [])

                # 验证必要字段
                if not group_name:
                    raise ValueError("变量组必须包含name字段")
                if not values:
                    raise ValueError(f"变量组 '{group_name}' 必须包含values字段")

                # 生成变量名并添加到网络
                group_vars = []
                for i in range(start, end + 1):
                    if suffix:
                        var_name = f"{prefix}_{i}_{suffix}"
                    else:
                        var_name = f"{prefix}_{i}"

                    network._add_variable(var_name, values)
                    group_vars.append(var_name)

                # 记录组内变量
                group_dict[group_name] = group_vars

            # 2. 处理单独的变量
            individual_vars = data.get('individual_variables', {})
            for var_name, values in individual_vars.items():
                if var_name != "_comment":  # _comment是注释字段
                    network._add_variable(var_name, values)

            # 3. 处理边
            edges_data = data.get('edges', {})

            # 3.1 处理单个边
            single_edges = edges_data.get('single_edges', [])
            for edge in single_edges:
                from_name = edge.get('from')
                to_name = edge.get('to')

                if not from_name or not to_name:
                    raise ValueError("边定义必须包含'from'和'to'字段")

                network._add_edge(from_name, to_name)

            # 4. 验证网络是否为有效DAG并设置模型
            if network._is_valid_network():
                network._set_model()
            else:
                raise ValueError(f"贝叶斯网络结构数据不符合有向无环图特征")

            return network

        except FileNotFoundError:
            raise FileNotFoundError(f"文件 '{filename}' 不存在")
        except json.JSONDecodeError:
            raise ValueError(f"文件 '{filename}' 格式错误")
        except Exception as e:
            raise RuntimeError(f"加载贝叶斯网络结构数据时发生错误: {str(e)}")

    def print_bn_structure(self):
        """
        打印贝叶斯网络结构信息

        输出格式:
        节点: X_1
          父节点: []
          子节点: ['X_2', 'X_3']

        节点: X_2
          父节点: ['X_1']
          子节点: ['X_4']
        """
        try:
            all_nodes = self.get_all_variables_name()
            print(f"共加载 {len(all_nodes)} 个节点，节点关系如下：\n")

            # 按节点名排序输出
            for node in sorted(all_nodes):
                parents = self.get_parents(node)
                children = self.get_children(node)

                print(f"节点: {node}")
                print(f"  父节点: {parents if parents else '无'}")
                print(f"  子节点: {children if children else '无'}\n")

        except Exception as e:
            print(f"加载或处理网络时出错: {str(e)}")


def cpd_csv_trans_t1(file_path):
    """
    处理贝叶斯网络CPD文件，将变量名X_i修改为X_i_t1

    参数:
    - file_path: CPD文件的完整路径

    返回:
    - 新生成的文件路径

    功能:
    - 读取原始CPD文件
    - 将所有X_i格式的变量名改为X_i_t1
    - 保存为新文件

    示例:
    输入: X_1_cpd.csv
    输出: X_1_t1_cpd.csv
    """
    # 创建新文件名
    dir_name, file_name = os.path.split(file_path)
    base_name, ext = os.path.splitext(file_name)

    # 使用正则表达式替换：X_i -> X_i_t1
    new_file_name = re.sub(r'X_(\d+)', r'X_\1_t1', base_name) + ext
    new_file_path = os.path.join(dir_name, new_file_name)

    # 读取并处理文件内容
    processed_lines = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 跳过注释行
            if line.strip().startswith('#'):
                processed_lines.append(line)
                continue

            # 修改变量名：X_i -> X_i_t1
            modified_line = re.sub(r'X_(\d+)', r'X_\1_t1', line)
            processed_lines.append(modified_line)

    # 保存处理后的内容
    with open(new_file_path, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)

    return new_file_path
