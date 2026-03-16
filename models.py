import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SD_CFR_ValueNetwork(nn.Module):
    """
    SD-CFR 的单一价值网络 (Value Network)。
    用于预测在特定信息集下，采取各个合法动作的反事实遗憾值 (Advantage)。
    """

    def __init__(self, input_dim=10, output_dim=3):
        super(SD_CFR_ValueNetwork, self).__init__()
        # 按照论文附录 A，Leduc 游戏的网络为 3 层，每层 64 个单元
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)

        # 输出层：对应 3 个可能动作的 Advantage
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # 最后一层不加激活函数，因为遗憾值可以是负数
        return self.output(x)


def get_strategy_from_value_net(value_net, state_tensor, legal_actions):
    """
    通过遗憾匹配 (Regret Matching) 将预测的遗憾值转化为概率分布 [cite: 418]。

    参数:
    - value_net: 当前训练的价值网络
    - state_tensor: 当前状态的张量表示 (1, 10)
    - legal_actions: 当前合法的动作索引列表，如 [0, 1] 或 [0, 1, 2]

    返回:
    - strategy: 长度为 3 的 numpy 数组，表示采取每个动作的概率
    """
    # 确保网络处于评估模式
    value_net.eval()
    with torch.no_grad():
        # 获取网络输出并转为一维 numpy 数组
        advantages = value_net(state_tensor).squeeze(0).numpy()

    strategy = np.zeros(3, dtype=np.float32)
    positive_advantages = np.zeros(3, dtype=np.float32)

    # 仅考虑合法的动作
    for a in legal_actions:
        # 实现论文中的 x_+ = max(x, 0) [cite: 418]
        positive_advantages[a] = max(advantages[a], 0.0)

    sum_advantages = np.sum(positive_advantages)

    # 公式 (4): 如果正向遗憾值之和大于 0，则按比例分配概率 [cite: 418]
    if sum_advantages > 0:
        strategy = positive_advantages / sum_advantages
    else:
        # 如果没有正向遗憾值，启发式地选择 advantage 最高的合法动作 [cite: 420]
        max_adv = -float("inf")
        best_a = -1
        for a in legal_actions:
            if advantages[a] > max_adv:
                max_adv = advantages[a]
                best_a = a

        # 将最优动作概率设为 1.0
        if best_a != -1:
            strategy[best_a] = 1.0

    return strategy
