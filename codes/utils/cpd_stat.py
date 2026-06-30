import re
from pathlib import Path

import pandas as pd


def add_counts_to_cpd(samples_path, cpd_path):
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

    print(f"已生成包含计数列的 CPD 文件：{output_path}")


def get_s_cpd_csv_files(generated_bns_dir):
    """
    遍历 generated_bns 下的每个 bn_* 子目录，组织其 data CSV 和全局 ground_truth CPD 文件。

    参数:
        generated_bns_dir: generated_bns 目录的路径

    返回:
        字典: {bn_name: {"cpd": [cpd文件名], "data": [s文件名]}}
    """
    base = Path(generated_bns_dir).resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"目录不存在: {base}")

    result = {}
    bn_pattern = re.compile(r"^bn_.+")
    s_pattern = re.compile(r"^S\d+\.csv$", re.IGNORECASE)
    v_pattern = re.compile(r"^V\d+_cpd\.csv$", re.IGNORECASE)

    for bn_dir in base.iterdir():
        if not bn_dir.is_dir() or not bn_pattern.match(bn_dir.name):
            continue

        # 收集 data 目录下的 S数字.csv 完整路径
        data_dir = bn_dir / "data"
        s_paths = []
        if data_dir.is_dir():
            for csv_file in data_dir.rglob("*.csv"):
                if s_pattern.match(csv_file.name):
                    s_paths.append(str(csv_file.resolve()))  # 绝对路径字符串
            s_paths.sort()

        # 收集 ground_truth 目录下的 V数字_cpd.csv 完整路径
        gt_dir = bn_dir / "ground_truth"
        v_paths = []
        if gt_dir.is_dir():
            for csv_file in gt_dir.glob("*.csv"):
                if v_pattern.match(csv_file.name):
                    v_paths.append(str(csv_file.resolve()))
            v_paths.sort()
        else:
            print(f"警告: {bn_dir.name} 下没有 ground_truth 目录")

        result[bn_dir.name] = {"cpd": v_paths, "data": s_paths}

    return result

#
# if __name__ == "__main__":
#     base_path = "../generated_bns/"
#     csv_path_dict = get_s_cpd_csv_files(base_path)
#     for k, v in csv_path_dict.items():
#         for sample_data in v["data"]:
#             for cpd_csv in v["cpd"]:
#                 add_counts_to_cpd(sample_data, cpd_csv)
