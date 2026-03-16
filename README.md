# SD-CFR for Leduc Poker

这是一个基于 PyTorch 实现的 **单深度反事实遗憾最小化 (Single Deep CFR, SD-CFR)** 算法的原型机。本项目使用精简的 Leduc Hold'em Poker 作为环境，用于在本地快速验证算法的收敛性与代码管道的正确性，为后续扩展到大型限注德州扑克打下基础。

## Rules of Leduc Poker
![Leduc poker rules](Leduc poker rules.png)
## 核心算法特性

本实现严格参考了 Eric Steinberger 的论文 _Single Deep Counterfactual Regret Minimization_

[Single Deep Counterfactual Regret Minimization](Single Deep Counterfactual Regret Minimization.pdf)

并包含了以下工程级别的优化：

1. **彻底摒弃平均策略网络**：通过保存价值网络（Value Networks）的历史迭代快照池 $B^M$，在实际推理和打牌时执行**轨迹采样 (Trajectory-sampling)** 来等效替代平均策略。
2. **外部采样 (External Sampling)**：在博弈树遍历时，对手采取单一路径采样，自身采取全路径探索，大幅降低计算开销。
3. **蓄水池采样 (Reservoir Buffer)**：保证经验回放池满载后，整体数据依然服从历史均匀分布。
4. **梯度防爆的 Linear CFR**：在计算 MSE Loss 时，按照产生数据时的迭代轮次 $t$ 进行线性加权，并引入了 `Batch-Norm` 风格的加权归一化，解决了在 Adam 优化器下后期梯度爆炸的问题。

## 项目结构

* `env.py`: 无状态的 Leduc Poker 环境与张量状态编码器 (State Encoder)。
* `models.py`: 深度神经网络结构 (Value Network) 与遗憾匹配逻辑 (Regret Matching)。
* `buffer.py`: 基于蓄水池采样的内存回放池。
* `train.py`: 核心训练脚本，包含博弈树遍历逻辑与神经网络的反向传播，最终将网络快照存入 $B^M$。
* `evaluate.py`: 对战与评估脚本，展示了如何通过 $B^M$ 进行轨迹采样打牌。

## 运行环境

* Python 3.8+
* PyTorch 2.0+
* NumPy

## 快速开始

### 1. 训练模型

直接运行训练脚本。它会在本地进行蒙特卡洛机会采样并训练模型。

```bash
python train.py
```
