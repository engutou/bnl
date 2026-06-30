from pgmpy.estimators.StructureScore import K2, BDeu

class MonoPenaltyMixin:
    """
    Mixin 类，封装单调性惩罚的结构先验逻辑。
    必须与 StructureScore 的子类（如 K2, BDeu）一起使用。

    beta 的物理含义与取值依据：
    beta 是结构先验的精度参数，控制单调性约束相对于数据似然的强度。
    因为 BDeu/K2 对数得分通常在数百至数千量级，而每条违边的惩罚项 phi
    最大仅为 0.1 (当 alpha=0.1, P_mono≈0)，因此 beta 必须设置为较大值
    (例如 500) 才能使惩罚与似然在决策中产生竞争。敏感性实验表明 beta 在
    [100, 1000] 范围内表现稳定。
    """

    def __init__(self, data, beta=500, expert_knowledge=None, **kwargs):
        """
        参数
        ----------
        data : pd.DataFrame
            观测数据集，列名为变量名。
        beta : float, default=500
            先验精度，控制单调性约束的整体强度。
        expert_knowledge : list of tuple, optional
            专家知识列表，格式为 [(from, to, local_penalty), ...]。
            每个三元组的含义：
            - from (str)  : 父节点名称。
            - to (str)    : 子节点名称。
            - local_penalty (float) : 局部惩罚值，计算公式为
              max(0, alpha - P_mono(from -> to))，其中 alpha 是置信容忍度
              (通常取 0.1)，P_mono 是贝叶斯弱单调性检验输出的后验概率。
            当 P_mono >= alpha 时，local_penalty = 0，该边不受惩罚；
            当 P_mono < alpha 时，local_penalty = alpha - P_mono，
            违背越严重则惩罚越大。
        **kwargs : dict
            传递给评分基类（K2 或 BDeu）的其他参数。
        """
        # 将剩余参数原样传递给实际的评分基类
        super().__init__(data, **kwargs)
        self.beta = beta
        self.expert_knowledge = expert_knowledge or []

    def local_score(self, variable, parents):
        """
        计算带专家知识约束的局部评分。

        总得分 = 对数边际似然 + 对数结构先验
        其中对数结构先验 = -beta * sum_{e in parents} max(0, alpha - P_mono(e))

        参数
        ----------
        variable : str
            当前变量名。
        parents : list of str
            当前变量的父节点列表。

        返回
        -------
        float
            经过单调性惩罚调整后的局部结构评分。
        """
        # 1. 调用评分基类的原始对数边际似然
        base_score = super().local_score(variable, parents)

        # 2. 计算当前变量在当前父节点集合下，因违背单调性而产生的总惩罚
        #    遍历所有专家知识三元组 (from, to, local_penalty)
        #    仅当约束的目标是当前变量 且 约束的父节点在当前父节点集合中时，
        #    才累加该惩罚项
        penalty = 0.0
        for from_v, to_v, local_pen in self.expert_knowledge:
            if to_v == variable and from_v in parents:
                penalty += local_pen

        # 3. 总得分 = 对数边际似然 - beta * sum(phi)
        return base_score - self.beta * penalty


class MonoPenaltyK2(MonoPenaltyMixin, K2):
    """
    带单调性结构先验的 K2 评分函数。

    在 K2 对数边际似然的基础上，减去 beta * sum(max(0, alpha - P_mono))
    以惩罚违反弱单调性约束的边。

    用法示例：
    k2_score = MonoPenaltyK2(data, beta=500, expert_knowledge=expert_knowledge)
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=k2_score, black_list=hier_blacklist)
    """
    pass


class MonoPenaltyBDeu(MonoPenaltyMixin, BDeu):
    """
    带单调性结构先验的 BDeu 评分函数。

    在 BDeu 对数边际似然的基础上，减去 beta * sum(max(0, alpha - P_mono))
    以惩罚违反弱单调性约束的边。

    用法示例：
    bdeu_score = MonoPenaltyBDeu(data, beta=500, expert_knowledge=expert_knowledge, equivalent_sample_size=2.0)
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=bdeu_score, black_list=hier_blacklist)

    注意：
    equivalent_sample_size 是 BDeu 的等效样本大小 (默认值为 1.0)。
    该值越大，Dirichlet 先验越强，需要更多数据才能改变初始的无信息先验；
    该值越小，先验越弱，数据的影响越大。
    """
    pass