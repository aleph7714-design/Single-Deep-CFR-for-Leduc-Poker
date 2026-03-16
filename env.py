import random
import numpy as np
import torch


class LeducStateEncoder:
    def __init__(self):
        # 3 (private) + 4 (public) + 3 (pot size, p0 commit, p1 commit) = 10 维
        self.input_dim = 10
        self.num_actions = 3  # 0: Fold, 1: Call/Check, 2: Raise

    def encode(self, private_card, public_card, pot, p0_commit, p1_commit):
        """
        将当前博弈状态转化为一维 Tensor
        """
        state_vec = np.zeros(self.input_dim, dtype=np.float32)

        # 1. 编码底牌 (0:J, 1:Q, 2:K)
        state_vec[private_card] = 1.0

        # 2. 编码公共牌
        if public_card == -1:
            state_vec[3] = 1.0  # 还没发公共牌 (Pre-flop)
        else:
            state_vec[4 + public_card] = 1.0

        # 3. 编码下注信息 (进行缩放以帮助神经网络收敛)
        # Leduc 极限锅底通常在 30 左右
        state_vec[7] = pot / 30.0
        state_vec[8] = p0_commit / 15.0
        state_vec[9] = p1_commit / 15.0

        return torch.tensor(state_vec).unsqueeze(0)


class LeducEnv:
    def __init__(self):
        self.encoder = LeducStateEncoder()
        # 牌库：6张牌，包含 {0, 0, 1, 1, 2, 2}，代表 {J, J, Q, Q, K, K} 或者论文中的 {A, A, B, B, C, C} [cite: 667]
        self.deck = [0, 0, 1, 1, 2, 2]
        self.cards = []  # 保存当前局的牌: [p0_card, p1_card, public_card]

    def reset(self):
        """蒙特卡洛机会采样：洗牌并确定本局的所有牌"""
        random.shuffle(self.deck)
        self.cards = self.deck[:3]
        return ""  # 初始 history 为空字符串

    def get_turn(self, history):
        """根据历史推断当前该谁行动。Leduc 中通常每轮都是 P0 先行动。"""
        rounds = history.split("/")
        current_round = rounds[-1]
        return len(current_round) % 2

    def evaluate_history(self, history):
        """
        解析历史字符串，计算双方投入的筹码和当前状态。
        动作映射: 'f'=Fold(0), 'c'=Call/Check(1), 'r'=Raise(2)
        返回: is_terminal, p0_commit, p1_commit
        """
        # 初始 Ante 每人 1 个筹码 [cite: 667]
        p0_commit = 1.0
        p1_commit = 1.0

        rounds = history.split("/")

        # 解析第一轮 (Pre-flop) [cite: 668]
        preflop = rounds[0]
        p0_turn = True
        for action in preflop:
            if action == "c":
                # Call: 匹配对手筹码
                if p0_turn:
                    p0_commit = p1_commit
                else:
                    p1_commit = p0_commit
            elif action == "r":
                # Raise: 匹配对手筹码后，额外加注 2 个筹码 [cite: 673]
                if p0_turn:
                    p0_commit = p1_commit + 2.0
                else:
                    p1_commit = p0_commit + 2.0
            p0_turn = not p0_turn

        # 解析第二轮 (Flop) 如果有的话 [cite: 668]
        if len(rounds) > 1:
            flop = rounds[1]
            p0_turn = True
            for action in flop:
                if action == "c":
                    if p0_turn:
                        p0_commit = p1_commit
                    else:
                        p1_commit = p0_commit
                elif action == "r":
                    # Flop 轮每次加注 4 个筹码 [cite: 673]
                    if p0_turn:
                        p0_commit = p1_commit + 4.0
                    else:
                        p1_commit = p0_commit + 4.0
                p0_turn = not p0_turn

        # 判断是否到达终点
        is_terminal = False
        if history.endswith("f"):
            is_terminal = True
        elif (
            history.endswith("cc") or history.endswith("rc") or history.endswith("rrc")
        ):
            if len(rounds) == 2:  # Flop 轮结束，进入 Showdown
                is_terminal = True

        return is_terminal, p0_commit, p1_commit

    def get_legal_actions(self, history):
        """
        获取当前合法的动作列表: [0(Fold), 1(Call), 2(Raise)]
        """
        rounds = history.split("/")
        current_round = rounds[-1]

        # 如果已经结束，没有合法动作
        if history.endswith("f") or (
            len(rounds) == 2
            and (
                history.endswith("cc")
                or history.endswith("rc")
                or history.endswith("rrc")
            )
        ):
            return []

        legal_actions = [0, 1]  # 永远可以 Fold 和 Call(Check)

        # 每轮最多允许 2 次 Raise [cite: 673]
        raises_in_round = current_round.count("r")
        if raises_in_round < 2:
            legal_actions.append(2)

        return legal_actions

    def is_next_round(self, history):
        """判断是否需要发公共牌（进入下一轮）"""
        rounds = history.split("/")
        if len(rounds) == 1:
            if (
                history.endswith("cc")
                or history.endswith("rc")
                or history.endswith("rrc")
            ):
                return True
        return False

    def get_payoff(self, history):
        """
        计算终局收益（从 P0 的视角返回真实赢取的筹码，如果是负数说明 P0 输了）。
        """
        _, p0_commit, p1_commit = self.evaluate_history(history)
        pot = p0_commit + p1_commit

        # 情况 1: 有人弃牌 [cite: 670]
        if history.endswith("f"):
            turn = self.get_turn(history)  # 此时的 turn 是刚好没采取动作的那个人
            # 最后一个行动的人是弃牌者
            if turn == 1:  # P0 弃牌，P1 赢走 P0 的筹码
                return -p0_commit
            else:  # P1 弃牌，P0 赢走 P1 的筹码
                return p1_commit

        # 情况 2: 摊牌比大小 (Showdown)
        p0_card = self.cards[0]
        p1_card = self.cards[1]
        public_card = self.cards[2]

        p0_matches = p0_card == public_card
        p1_matches = p1_card == public_card

        # 规则：匹配公共牌的直接获胜 [cite: 675]
        if p0_matches and not p1_matches:
            return p1_commit
        elif p1_matches and not p0_matches:
            return -p0_commit

        # 否则，比单牌大小 [cite: 676]
        if p0_card > p1_card:
            return p1_commit
        elif p1_card > p0_card:
            return -p0_commit
        else:
            return 0.0  # 平局，退回筹码

    def get_state_tensor(self, history, player):
        """
        供价值网络使用：将当前状态转化为 Tensor
        """
        _, p0_commit, p1_commit = self.evaluate_history(history)
        pot = p0_commit + p1_commit
        private_card = self.cards[player]

        rounds = history.split("/")
        if len(rounds) == 1:
            public_card = -1  # Pre-flop
        else:
            public_card = self.cards[2]  # Flop

        return self.encoder.encode(private_card, public_card, pot, p0_commit, p1_commit)
