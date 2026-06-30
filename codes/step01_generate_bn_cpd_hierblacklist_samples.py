import os

from utils.build_model_from_data import MyBayesianNetwork
from utils.generate_bn_network_cpd import generate_bayesian_network
from utils.generate_hierarchical_blacklist import generate_hierarchical_blacklist
from utils.generate_samples import SampleGenerator
from utils.prob_monotonic_binary import generate_monotonicity_blacklist

if __name__ == "__main__":
    # num_nodes, num_edges, count = 24, 35, 10
    num_nodes, num_edges, count = 12, 16, 10
    sample_count_list = [200, 500, 1000, 2000, 5000, 10000]
    prob_range = (0.05, 0.2)
    for idx in range(count):
        # count: 随机生成作为ground-truth的贝叶斯网络多次，每次是一个独立的实验
        # 生成的数据存放在ground_truth目录下
        print("*" * 50, "开始生成贝叶斯网络及cpt表！", "*" * 50)
        if idx + 1 <= 9:
            folder_name = f"../experiments_N{num_nodes}_E{num_edges}/generated_bns/bn_N{num_nodes}_E{num_edges}_0{idx + 1}/ground_truth/"
        else:
            folder_name = f"../experiments_N{num_nodes}_E{num_edges}/generated_bns/bn_N{num_nodes}_E{num_edges}_{idx + 1}/ground_truth/"
        generate_bayesian_network(
            num_nodes=num_nodes,
            target_edges=num_edges,
            prob_range=prob_range,
            effect_strength_range=(0.3, 0.5),
            max_parents=3,
            random_seed=44 + idx,
            folder_name=folder_name)

        # 设定层级约束专家知识
        print("#" * 24, "开始生成层级约束专家知识", "#" * 24)
        dag_json_path = os.path.join(folder_name, 'dag.json')
        generate_hierarchical_blacklist(topo_file_path=dag_json_path)

        # 生成对应的样本数据
        for sample_count in sample_count_list:
            print("#" * 24, f"开始生成 {sample_count} 样本数据", "#" * 24)
            network = MyBayesianNetwork.load_model_from_json_file(dag_json_path)

            # 2. 创建样本生成器
            generator = SampleGenerator(network)

            # 4. 加载CPD
            cpds_loaded = generator.load_cpds()
            # 5. 验证模型
            if not generator.validate_model():
                print("模型验证失败，无法生成样本")
            else:
                # 6. 生成样本数据
                try:
                    # 使用pgmpy的simulate方法生成样本
                    samples = generator.generate_samples(num_samples=sample_count, method_order=[1])
                    # 8. 保存样本数据，更新cpd的统计和频率
                    samples_csv_path = generator.save_samples_update_cpd(samples, F"S{sample_count}")
                    # 生成monotonicity blacklist数据
                    generate_monotonicity_blacklist(samples_csv_path)


                except Exception as e:
                    print(f"生成样本数据时发生错误: {e}")
