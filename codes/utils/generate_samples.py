import os

import numpy as np
import pandas as pd


class SampleGenerator:
    """贝叶斯网络样本生成器"""

    def __init__(self, network):
        """
        初始化样本生成器

        参数:
        - network: MyBayesianNetwork对象实例
        """
        self.network = network
        self.csv_path_list = []

    def load_cpds(self, data_dir=None):
        """
        为贝叶斯网络的所有节点加载CPD

        参数:
        - data_dir: CPD文件目录，如果为None则使用network.data_dir
        """
        # 在 load_cpds 函数开头添加

        if data_dir is None:
            data_dir = self.network.data_dir

        cpds_loaded = []
        for var_name in self.network.get_all_variables_name():
            csv_filename = f"{var_name}_cpd.csv"
            csv_path = os.path.join(data_dir, csv_filename)
            self.csv_path_list.append(csv_path)
            try:
                self.network.add_cpd_from_csv(csv_path)
                cpds_loaded.append(var_name)
            except Exception as e:
                print(f"加载变量 {var_name} 的CPD失败: {e}")

        print(f"✓ 成功加载 {len(cpds_loaded)} 个变量的CPD")
        return cpds_loaded

    def validate_model(self):
        """
        验证模型是否完全指定

        返回:
        - True: 验证成功
        - False: 验证失败
        """
        try:
            # 有向无环图(DAG)验证：确保网络不存在循环依赖
            # 节点完整性：确保所有在边中提到的节点都已定义
            # 父节点一致性：确保CPD中定义的父节点与网络结构中的父节点一致
            is_valid = self.network.model.check_model()
            if is_valid:
                print("✓ 模型验证成功：网络结构和CPD一致")
                return True
            else:
                print("✗ 模型验证失败：网络结构或CPD存在问题")
                return False
        except Exception as e:
            print(f"模型验证时出错：{e}")
            return False

    def generate_samples_method1_simulate(self, num_samples, show_progress=True):
        """
        方法1：使用pgmpy的simulate方法生成样本

        参数:
        - num_samples: 样本数量
        - show_progress: 是否显示进度条

        返回:
        - pandas DataFrame: 生成的样本数据
        - str: 错误信息，成功时为None
        """
        if not hasattr(self.network.model, 'simulate'):
            return None, "模型不支持simulate方法"

        try:
            # 使用simulate方法生成样本
            samples = self.network.model.simulate(
                n_samples=num_samples,
                show_progress=show_progress
            )

            print(f"✓ 成功生成 {len(samples)} 条样本数据 (方法1: simulate)")
            return samples, None

        except Exception as e:
            error_msg = f"simulate方法生成样本失败: {e}"
            print(f"✗ {error_msg}")
            return None, error_msg

    def generate_samples_method2_forward_sampling(self, num_samples=10000, show_progress=True):
        """
        方法2：使用贝叶斯网络的前向采样算法生成样本

        参数:
        - num_samples: 样本数量
        - show_progress: 是否显示进度条

        返回:
        - pandas DataFrame: 生成的样本数据
        - str: 错误信息，成功时为None
        """
        print("尝试使用前向采样算法生成样本...")

        try:
            from pgmpy.sampling import BayesianModelSampling

            inference = BayesianModelSampling(self.network.model)

            # 使用前向采样算法
            samples = inference.forward_sample(
                size=num_samples,
                show_progress=show_progress
            )

            print(f"✓ 成功生成 {len(samples)} 条样本数据 (方法2: 前向采样)")
            return samples, None

        except Exception as e:
            error_msg = f"前向采样算法生成样本失败: {e}"
            print(f"✗ {error_msg}")
            return None, error_msg

    def generate_samples_method3_custom_sampling(self, num_samples=10000):
        """
        方法3：使用自定义的随机抽样算法生成样本

        参数:
        - num_samples: 样本数量

        返回:
        - pandas DataFrame: 生成的样本数据
        - str: 错误信息，成功时为None
        """
        print("尝试使用自定义随机抽样算法生成样本...")

        try:
            # 获取所有变量
            variables = self.network.get_all_variables_name()

            # 创建空的DataFrame
            samples = pd.DataFrame(columns=variables)

            # 获取每个变量的CPD
            cpds = {}
            for var in variables:
                # 获取变量的CPD
                cpd = self.network.model.get_cpds(var)
                cpds[var] = cpd

            # 生成样本
            for i in range(num_samples):
                if i % 1000 == 0 and i > 0:
                    print(f"  已生成 {i}/{num_samples} 条样本...")

                sample = {}

                # 按拓扑顺序生成变量值
                # 首先找出根节点（没有父节点的节点）
                for var in variables:
                    parents = self.network.get_parents(var)
                    if not parents:  # 根节点
                        # 根据先验概率分布生成值
                        cpd = cpds[var]
                        probs = cpd.values
                        # 随机选择值
                        values = self.network.get_variable_values(var)
                        # 确保概率和为1
                        probs_flat = probs.flatten()
                        probs_flat = probs_flat / probs_flat.sum()
                        chosen_value = np.random.choice(values, p=probs_flat)
                        sample[var] = chosen_value

                # 然后按拓扑顺序生成其他节点
                # 这里简化处理：多次迭代直到所有变量都有值
                max_iterations = len(variables) * 2
                for _ in range(max_iterations):
                    for var in variables:
                        if var in sample:
                            continue

                        parents = self.network.get_parents(var)
                        # 检查所有父节点是否都有值
                        if all(parent in sample for parent in parents):
                            cpd = cpds[var]
                            values = self.network.get_variable_values(var)

                            # 获取父节点的取值组合对应的概率分布
                            if parents:
                                # 尝试从CPD中获取正确的概率分布
                                try:
                                    # 获取CPD的状态名称
                                    state_names = cpd.state_names

                                    # 找到当前父节点取值对应的列索引
                                    parent_values = tuple(sample[parent] for parent in parents)

                                    # 遍历所有可能的父节点取值组合，找到匹配的
                                    found = False
                                    for col_idx, evidence_combo in enumerate(cpd.get_evidence()):
                                        if evidence_combo == parent_values:
                                            # 找到对应的概率分布
                                            probs = cpd.values[:, col_idx]
                                            probs = probs / probs.sum()  # 归一化
                                            chosen_value = np.random.choice(values, p=probs)
                                            sample[var] = chosen_value
                                            found = True
                                            break

                                    if not found:
                                        # 如果没找到精确匹配，使用均匀分布
                                        chosen_value = np.random.choice(values)
                                        sample[var] = chosen_value

                                except Exception:
                                    # 出现异常时使用均匀分布
                                    chosen_value = np.random.choice(values)
                                    sample[var] = chosen_value
                            else:
                                # 无父节点，使用先验概率
                                probs = cpd.values
                                probs_flat = probs.flatten()
                                probs_flat = probs_flat / probs_flat.sum()
                                chosen_value = np.random.choice(values, p=probs_flat)
                                sample[var] = chosen_value

                    # 检查是否所有变量都有值
                    if len(sample) == len(variables):
                        break

                # 添加到样本数据
                samples.loc[i] = [sample[var] for var in variables]

            print(f"✓ 成功生成 {len(samples)} 条样本数据 (方法3: 自定义抽样)")
            return samples, None

        except Exception as e:
            error_msg = f"自定义随机抽样算法生成样本失败: {e}"
            print(f"✗ {error_msg}")
            return None, error_msg

    def generate_samples(self, num_samples, method_order=None):
        """
        综合生成样本，按指定顺序尝试不同的方法

        参数:
        - num_samples: 样本数量
        - method_order: 方法尝试顺序，例如[1, 2, 3]

        返回:
        - pandas DataFrame: 生成的样本数据
        """
        if method_order is None:
            method_order = [1, 2, 3]  # 默认尝试顺序

        samples = None
        error_messages = []

        for method_num in method_order:
            if method_num == 1:
                samples, error = self.generate_samples_method1_simulate(num_samples)
            elif method_num == 2:
                samples, error = self.generate_samples_method2_forward_sampling(num_samples)
            elif method_num == 3:
                samples, error = self.generate_samples_method3_custom_sampling(num_samples)
            else:
                error = f"未知的方法编号: {method_num}"

            if error:
                error_messages.append(f"方法{method_num}: {error}")
            else:
                break  # 成功生成样本，退出循环

        if samples is None:
            print("\n✗ 所有方法都失败了:")
            for msg in error_messages:
                print(f"  - {msg}")
            raise RuntimeError("无法生成样本数据")

        return samples

    def save_samples_update_cpd(self, samples, filename):
        """
        保存样本数据到CSV文件, 同时更新cpd的统计和频率

        参数:
        - samples: pandas DataFrame样本数据
        - filename: 保存文件名，如果为None则自动生成

        返回:
        - str: 保存的文件路径
        """
        base_dir = self.network.data_dir.replace("ground_truth", f"data\\{filename}")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        samples_file = os.path.join(base_dir, filename + ".csv")
        samples.to_csv(samples_file, index=False)

        print(f"✓ 样本数据已保存到: {samples_file}")

        # 针对样本数据，统计cpd中每行的数量和频率
        print(f"✓ 根据样本与CPD表，添加cpd表每行在样本中的次数与频率")
        for cpd_csv in self.csv_path_list:
            self.add_counts_to_cpd(samples_file, cpd_csv)
        return samples_file

    def add_counts_to_cpd(self, samples_path, cpd_path):
        """
        从样本数据中统计每个父节点组合下目标变量的计数，并将计数作为新列添加到 CPD 文件中。
        支持无父节点的情况（parents 为空）。
        """
        # 1. 读取 CPD 文件，将 parents 列作为字符串读取
        cpd = pd.read_csv(cpd_path, sep=';', dtype={'parents': str})
        target_var = cpd.iloc[0]['target_variable']
        parents_str = cpd.iloc[0]['parents']

        # 处理空 parents 的情况
        if pd.isna(parents_str) or str(parents_str).strip() == '':
            parents = []
        else:
            parents = [p.strip() for p in str(parents_str).split(',') if p.strip()]

        # 2. 读取样本数据
        samples = pd.read_csv(samples_path)
        # 检查必要列是否存在（如果 parents 非空）
        for col in parents + [target_var]:
            if col not in samples.columns:
                raise ValueError(f"样本文件中缺少列: {col}")

        # 3. 统计计数
        count_col_0 = f'{target_var}=0_COUNT'
        count_col_1 = f'{target_var}=1_COUNT'

        if parents:
            # 有父节点：按组合分组统计
            counts = samples.groupby(parents)[target_var].value_counts().unstack(fill_value=0)
            # 确保列包含 0 和 1（防止某个值缺失）
            counts = counts.reindex(columns=[0, 1], fill_value=0)
            counts.columns = [count_col_0, count_col_1]
            counts = counts.reset_index()
        else:
            # 无父节点：全局统计
            value_counts = samples[target_var].value_counts()
            count_0 = value_counts.get(0, 0)
            count_1 = value_counts.get(1, 0)
            counts = pd.DataFrame({count_col_0: [count_0], count_col_1: [count_1]})

        # 4. 重命名 CPD 中的父节点列名（仅当有父节点时）
        if parents:
            rename_map = {}
            for p in parents:
                # 匹配 "p(parent_i)" 或 "p" 形式的列名
                matching = [col for col in cpd.columns if col.startswith(p + '(') or col == p]
                if matching:
                    rename_map[matching[0]] = p
            cpd.rename(columns=rename_map, inplace=True)

        # 5. 合并计数到 CPD 表
        if parents:
            cpd_with_counts = cpd.merge(counts, on=parents, how='left')
        else:
            # 无父节点：直接添加计数列（CPD 表中只有一行）
            cpd_with_counts = cpd.copy()
            cpd_with_counts[count_col_0] = counts[count_col_0].iloc[0]
            cpd_with_counts[count_col_1] = counts[count_col_1].iloc[0]

        # 填充可能的缺失值并转为整数
        cpd_with_counts[count_col_0] = cpd_with_counts[count_col_0].fillna(0).astype(int)
        cpd_with_counts[count_col_1] = cpd_with_counts[count_col_1].fillna(0).astype(int)

        # 6. 重命名为最终列名（带 (COUNT) 后缀）
        final_col_0 = f'{target_var}=0(COUNT)'
        final_col_1 = f'{target_var}=1(COUNT)'
        cpd_with_counts[final_col_0] = cpd_with_counts[count_col_0]
        cpd_with_counts[final_col_1] = cpd_with_counts[count_col_1]
        cpd_with_counts.drop([count_col_0, count_col_1], axis=1, inplace=True)

        # 7. 添加概率列
        # 7. 添加概率列，保留6位小数
        total = cpd_with_counts[final_col_0] + cpd_with_counts[final_col_1]
        prob_0_col = f'P({target_var}=0)'
        prob_1_col = f'P({target_var}=1)'
        # 避免除零：总数为0时概率为0，并保留6位小数
        cpd_with_counts[prob_0_col] = (cpd_with_counts[final_col_0] / total).fillna(0).round(6)
        cpd_with_counts[prob_1_col] = (cpd_with_counts[final_col_1] / total).fillna(0).round(6)

        # 8. 保存结果
        # 'D:\\pro_code\\DBN\\experiments\\generated_bns\\bn_N24_E35_01\\data\\S10000\\S10000.csv'
        # 'D:\\pro_code\\DBN\\experiments\\generated_bns\\bn_N24_E35_01\\ground_truth\\V10_cpd.csv'
        cpd_file_name = cpd_path.split("\\")[-1]
        output_path = samples_path.replace(".csv", "_" + cpd_file_name)
        cpd_with_counts.to_csv(output_path, sep=';', index=False)

        # print(f"已生成包含计数列的 CPD 文件：{output_path}")

    def print_sample_statistics(self, samples):
        """
        打印样本数据统计信息

        参数:
        - samples: pandas DataFrame样本数据
        """
        print(f"\n样本数据统计信息:")
        print(f"  总样本数: {len(samples)}")
        print(f"  变量数量: {len(samples.columns)}")


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

# if __name__ == '__main__':
#     base_path = "./generated_bns/"
#     json_path_list = find_json_files(base_path)
#     sample_count_list = [100, 200, 500, 1000, 5000, 10000]
#     for json_path in json_path_list:
#         for sample_count in sample_count_list:
#             print("*" * 33, "正在加载贝叶斯网络", "*" * 33)
#             network = MyBayesianNetwork.load_model_from_json_file(json_path)
#
#             # 2. 创建样本生成器
#             generator = SampleGenerator(network)
#
#             # 3. 检查并生成CPD文件（如果没有的话）
#             cpd_files_exist = True
#             for var_name in network.get_all_variables_name():
#                 csv_filename = f"{var_name}_cpd.csv"
#                 csv_path = os.path.join(network.data_dir, csv_filename)
#                 if not os.path.exists(csv_path):
#                     cpd_files_exist = False
#                     raise (f"CPD文件不存在: {csv_path}")
#
#             # 4. 加载CPD
#             cpds_loaded = generator.load_cpds()
#
#             # 5. 验证模型
#             if not generator.validate_model():
#                 print("模型验证失败，无法生成样本")
#             else:
#                 # 6. 生成样本数据
#                 print(f"\n正在生成 {sample_count} 条样本数据...")
#                 try:
#                     # 使用综合方法生成样本，按顺序尝试方法1、2、3
#                     # 1、使用pgmpy的simulate方法生成样本
#                     # 2、使用贝叶斯网络的前向采样算法生成样本
#                     # 3、使用自定义的随机抽样算法生成样本
#                     samples = generator.generate_samples(num_samples=sample_count, method_order=[1])
#
#                     # 7. 打印样本统计信息
#                     generator.print_sample_statistics(samples)
#
#                     # 8. 保存样本数据
#                     generator.save_samples(samples, F"S{sample_count}")
#
#                     print("\n✓ 样本生成完成！")
#
#                 except Exception as e:
#                     print(f"生成样本数据时发生错误: {e}")
