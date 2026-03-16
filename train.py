import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

# 导入我们之前写好的模块
from env import LeducEnv
from models import SD_CFR_ValueNetwork, get_strategy_from_value_net
from buffer import ReservoirBuffer

# --- 超参数设置 ---
# M4 快速测试版超参数 (如果放到 RTX 3070 训练完整版，请参考下方注释)
ITERATIONS = 50  # 论文标准版: 1000+
TRAVERSALS_PER_ITER = 100  # 论文标准版: 1500
BATCH_SIZE = 256  # 论文标准版: 2048
UPDATE_STEPS = 100  # 论文标准版: 750 (每次迭代网络优化的步数)
BUFFER_CAPACITY = 200000  # 论文标准版: 1,000,000

ACTION_MAP = {0: "f", 1: "c", 2: "r"}


def traverse(env, history, traverser, iteration, nets, buffers):
    """
    外部采样遍历器 (External Sampling Traverser)
    """
    # 1. 终局判断
    is_terminal, _, _ = env.evaluate_history(history)
    if is_terminal:
        payoff = env.get_payoff(history)
        # 返回对 Traverser 的收益
        return payoff if traverser == 0 else -payoff

    # 2. 轮次转换判断 (加上 '/' 表示进入 Flop 圈)
    if env.is_next_round(history) and not history.endswith("/"):
        return traverse(env, history + "/", traverser, iteration, nets, buffers)

    # 3. 获取当前节点信息
    turn = env.get_turn(history)
    state_tensor = env.get_state_tensor(history, turn)
    legal_actions = env.get_legal_actions(history)

    # 极罕见情况防错：如果没有合法动作但还没被判定为terminal
    if not legal_actions:
        return 0.0

    # 4. 从当前价值网络获取策略
    strategy = get_strategy_from_value_net(nets[turn], state_tensor, legal_actions)

    # ==========================================
    # 分支 A: 当前节点属于对手 (Opponent) -> 采样单一动作
    # ==========================================
    if turn != traverser:
        # 按照策略概率分布采样一个动作
        a = np.random.choice(3, p=strategy)
        action_str = ACTION_MAP[a]
        # 递归下去，直接返回采样路径的收益
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
            # 递归计算采取动作 a 后的期望收益
            util = traverse(
                env, history + action_str, traverser, iteration, nets, buffers
            )
            action_utilities[a] = util
            expected_utility += strategy[a] * util

        # 计算每个动作的瞬时反事实遗憾值 (Regret)
        regrets = np.zeros(3, dtype=np.float32)
        for a in legal_actions:
            # 遗憾值 = 采取该动作的收益 - 按当前策略的期望收益
            regrets[a] = action_utilities[a] - expected_utility

        # 将产生的数据 (状态, 遗憾值, 当前迭代次数t) 存入该玩家的蓄水池 Memory Buffer
        buffers[traverser].add(state_tensor, regrets, iteration)

        # 向上层节点返回期望收益
        return expected_utility


def train_value_network(net, buffer, optimizer, iteration):
    """
    使用经验回放池中的数据训练价值网络。
    引入 Linear CFR 机制：损失函数由 t (产生数据时的迭代轮次) 加权。
    """
    # 如果 Buffer 里数据还不够一个 Batch，先跳过训练
    if len(buffer) < BATCH_SIZE:
        return 0.0

    net.train()
    total_loss = 0.0

    for _ in range(UPDATE_STEPS):
        states, target_regrets, weights = buffer.sample(BATCH_SIZE)

        optimizer.zero_grad()
        predicted_regrets = net(states)

        # 核心：Linear CFR 加权的均方误差 (MSE) 损失
        # weight 即为加入 buffer 时的 iteration t，越新的数据权重越高
        # 将 weight 进行 Batch 内归一化，防止随着 t 增大导致梯度爆炸
        normalized_weights = weights / (weights.mean() + 1e-8)
        loss = (normalized_weights * (predicted_regrets - target_regrets) ** 2).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / UPDATE_STEPS


if __name__ == "__main__":
    print("🤖 启动 SD-CFR 训练 (Leduc Poker 快速验证版)...")

    # 初始化环境
    env = LeducEnv()

    # 为 P0 和 P1 分别初始化价值网络、优化器和缓冲区
    nets = {0: SD_CFR_ValueNetwork(), 1: SD_CFR_ValueNetwork()}

    # 论文默认使用 Adam 优化器 [cite: 627]
    optimizers = {
        0: optim.Adam(nets[0].parameters(), lr=0.001),
        1: optim.Adam(nets[1].parameters(), lr=0.001),
    }

    buffers = {0: ReservoirBuffer(BUFFER_CAPACITY), 1: ReservoirBuffer(BUFFER_CAPACITY)}

    start_time = time.time()

    # --- 主训练循环 ---
    for t in range(1, ITERATIONS + 1):
        # 1. 外部采样数据生成阶段 (Traversals)
        # Deep CFR/SD-CFR 是交替更新的 (Alternating Updates) [cite: 407]
        for traverser in [0, 1]:
            for _ in range(TRAVERSALS_PER_ITER):
                # 每一次 Traversal 洗牌一次 (Monte Carlo Chance Sampling)
                history = env.reset()
                traverse(env, history, traverser, t, nets, buffers)

        # 2. 神经网络训练阶段 (Training)
        loss_p0 = train_value_network(nets[0], buffers[0], optimizers[0], t)
        loss_p1 = train_value_network(nets[1], buffers[1], optimizers[1], t)

        # 打印训练进度
        if t % 5 == 0 or t == 1:
            elapsed = time.time() - start_time
            print(
                f"Iter {t:3d}/{ITERATIONS} | "
                f"P0 Loss: {loss_p0:.4f} | P1 Loss: {loss_p1:.4f} | "
                f"P0 Buffer: {len(buffers[0])} | Time: {elapsed:.1f}s"
            )

    print("\n✅ 训练完成！")
    print(
        "M4 本地原型已跑通。如果要评估策略，可以编写一个对局脚本，直接调用 `get_strategy_from_value_net`。"
    )
