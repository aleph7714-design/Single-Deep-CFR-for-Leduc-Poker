import torch
import torch.optim as optim
import numpy as np
import time
import copy
import os

# 导入我们之前写好的模块
from env import LeducEnv
from models import SD_CFR_ValueNetwork, get_strategy_from_value_net
from buffer import ReservoirBuffer

# --- 超参数设置 (MacBook M4 原型测试版) ---
ITERATIONS = 50  # 迭代总轮数 (完整训练建议 1000+)
TRAVERSALS_PER_ITER = 100  # 每轮外部采样的遍历次数
BATCH_SIZE = 256  # 神经网络训练的 Batch 大小
UPDATE_STEPS = 100  # 每轮迭代网络优化的步数
BUFFER_CAPACITY = 200000  # 蓄水池内存上限

ACTION_MAP = {0: "f", 1: "c", 2: "r"}


def traverse(env, history, traverser, iteration, nets, buffers):
    """
    外部采样遍历器 (External Sampling Traverser)
    """
    # 1. 终局判断
    is_terminal, _, _ = env.evaluate_history(history)
    if is_terminal:
        payoff = env.get_payoff(history)
        return payoff if traverser == 0 else -payoff

    # 2. 轮次转换判断
    if env.is_next_round(history) and not history.endswith("/"):
        return traverse(env, history + "/", traverser, iteration, nets, buffers)

    # 3. 获取当前节点信息
    turn = env.get_turn(history)
    state_tensor = env.get_state_tensor(history, turn)
    legal_actions = env.get_legal_actions(history)

    if not legal_actions:
        return 0.0

    # 4. 从当前价值网络获取策略
    strategy = get_strategy_from_value_net(nets[turn], state_tensor, legal_actions)

    # ==========================================
    # 分支 A: 当前节点属于对手 (Opponent) -> 采样单一动作
    # ==========================================
    if turn != traverser:
        a = np.random.choice(3, p=strategy)
        action_str = ACTION_MAP[a]
        return traverse(env, history + action_str, traverser, iteration, nets, buffers)

    # ==========================================
    # 分支 B: 当前节点属于遍历者 (Traverser) -> 探索所有合法动作
    # ==========================================
    else:
        expected_utility = 0.0
        action_utilities = np.zeros(3, dtype=np.float32)

        # 遍历所有合法的动作
        for a in legal_actions:
            action_str = ACTION_MAP[a]
            util = traverse(
                env, history + action_str, traverser, iteration, nets, buffers
            )
            action_utilities[a] = util
            expected_utility += strategy[a] * util

        # 计算遗憾值 (Regret)
        regrets = np.zeros(3, dtype=np.float32)
        for a in legal_actions:
            regrets[a] = action_utilities[a] - expected_utility

        # 存入蓄水池
        buffers[traverser].add(state_tensor, regrets, iteration)

        return expected_utility


def train_value_network(net, buffer, optimizer, iteration):
    """
    使用经验回放池中的数据训练价值网络 (带权重归一化修复)
    """
    if len(buffer) < BATCH_SIZE:
        return 0.0

    net.train()
    total_loss = 0.0

    for _ in range(UPDATE_STEPS):
        states, target_regrets, weights = buffer.sample(BATCH_SIZE)

        optimizer.zero_grad()
        predicted_regrets = net(states)

        # Linear CFR 加权 + 权重归一化 (防止 Adam 梯度爆炸)
        normalized_weights = weights / (weights.mean() + 1e-8)
        loss = (normalized_weights * (predicted_regrets - target_regrets) ** 2).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / UPDATE_STEPS


if __name__ == "__main__":
    print("🤖 启动 SD-CFR 训练 (Leduc Poker 快速验证版)...")

    env = LeducEnv()

    nets = {0: SD_CFR_ValueNetwork(), 1: SD_CFR_ValueNetwork()}

    optimizers = {
        0: optim.Adam(nets[0].parameters(), lr=0.001),
        1: optim.Adam(nets[1].parameters(), lr=0.001),
    }

    buffers = {0: ReservoirBuffer(BUFFER_CAPACITY), 1: ReservoirBuffer(BUFFER_CAPACITY)}

    # 核心新增：初始化历史网络池 B^M
    B_M = {0: [], 1: []}

    start_time = time.time()

    for t in range(1, ITERATIONS + 1):
        # 1. 数据生成阶段
        for traverser in [0, 1]:
            for _ in range(TRAVERSALS_PER_ITER):
                history = env.reset()
                traverse(env, history, traverser, t, nets, buffers)

        # 2. 网络训练阶段
        loss_p0 = train_value_network(nets[0], buffers[0], optimizers[0], t)
        loss_p1 = train_value_network(nets[1], buffers[1], optimizers[1], t)

        # 3. 核心新增：将本轮训练好的大脑快照存入 B^M
        # 使用 state_dict 并在 cpu 上保存，极大地节省内存
        B_M[0].append(copy.deepcopy(nets[0].state_dict()))
        B_M[1].append(copy.deepcopy(nets[1].state_dict()))

        if t % 5 == 0 or t == 1:
            elapsed = time.time() - start_time
            print(
                f"Iter {t:3d}/{ITERATIONS} | "
                f"P0 Loss: {loss_p0:.4f} | P1 Loss: {loss_p1:.4f} | "
                f"Buffer: {len(buffers[0])} | Time: {elapsed:.1f}s"
            )

    print("\n✅ 训练完成！")

    # 将完整的历史网络池持久化保存到硬盘
    save_path = "sdcfr_models_BM.pth"
    torch.save(B_M, save_path)
    print(
        f"历史网络池已成功保存至 {save_path}。现在可以运行 evaluate.py 来进行对战评估了！"
    )
