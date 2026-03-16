import torch
import numpy as np
from env import LeducEnv
from models import get_strategy_from_value_net


def sample_network_from_BM(B_M_player):
    """
    核心机制：轨迹采样 (Trajectory-sampling)
    从历史网络池 B^M 中选择一个价值网络，采样权重与迭代次数 t 成正比。
    """
    T = len(B_M_player)
    # 生成权重列表: [1, 2, 3, ..., T]
    weights = np.arange(1, T + 1, dtype=np.float32)

    # 归一化为概率分布
    probabilities = weights / np.sum(weights)

    # 按照概率分布随机抽取一个索引
    # 越靠后（t 越大）的网络被抽中的概率越高
    sampled_index = np.random.choice(T, p=probabilities)

    # 返回被选中的那个价值网络
    return B_M_player[sampled_index]


def play_one_hand(env, B_M_p0, B_M_p1):
    """
    使用抽样出的网络打一局完整的牌
    """
    # 1. 游戏开始时，双方各自按照权重 t 从 B^M 中抽取一个网络
    # 这整局游戏都会固定使用这两个抽出来的网络
    net_p0 = sample_network_from_BM(B_M_p0)
    net_p1 = sample_network_from_BM(B_M_p1)

    # 将网络设置为评估模式
    net_p0.eval()
    net_p1.eval()

    history = env.reset()
    print(f"发牌完毕！P0底牌:{env.cards[0]}, P1底牌:{env.cards[1]}")

    while True:
        is_terminal, p0_commit, p1_commit = env.evaluate_history(history)
        if is_terminal:
            payoff = env.get_payoff(history)
            print(f"💰 终局！历史: {history} | P0收益: {payoff}")
            return payoff

        if env.is_next_round(history) and not history.endswith("/"):
            history += "/"
            print(f"--- 进入 Flop 圈 (公共牌: {env.cards[2]}) ---")
            continue

        turn = env.get_turn(history)
        state_tensor = env.get_state_tensor(history, turn)
        legal_actions = env.get_legal_actions(history)

        # 2. 整局游戏 (Trajectory) 都使用这一个选定的网络提供的策略
        active_net = net_p0 if turn == 0 else net_p1
        strategy = get_strategy_from_value_net(active_net, state_tensor, legal_actions)

        # 根据算出的概率分布随机选择一个动作
        action_idx = np.random.choice(3, p=strategy)
        action_map = {0: "fold", 1: "call/check", 2: "raise"}
        action_char = {0: "f", 1: "c", 2: "r"}[action_idx]

        print(f"玩家 P{turn} 动作分布: {strategy} -> 选择了: {action_map[action_idx]}")
        history += action_char


if __name__ == "__main__":
    print("加载历史网络池 B_M...")
    B_M = torch.load("sdcfr_models_BM.pth")

    env = LeducEnv()

    # 跑 5 局看看效果
    for i in range(5):
        print(f"\n================ 第 {i+1} 局 ================")
        play_one_hand(env, B_M[0], B_M[1])
