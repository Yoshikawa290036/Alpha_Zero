import random
import math

class State:
    def __init__(self, pieces=None, enemy_pieces=None):
        self.pieces = pieces if pieces != None else [0] * 25
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [0]*25

    def piece_count(self, pieces):
        count = 0
        for piece in pieces:
            if piece == 1:
                count += 1
        return count

    def is_lose(self):
        # 横方向
        for i in range(5):
            for j in range(5):
                index = i*5+j
                if self.enemy_pieces[index] == 0:
                    break
            else:
                return True
        # 縦方向
        for j in range(5):
            for i in range(5):
                index = i*5+j
                if self.enemy_pieces[index] == 0:
                    break
            else:
                return True
        # 左斜め
        for ij in range(0, 25, 6):
            if self.enemy_pieces[ij] == 0:
                break
        else:
            return True
        # 右斜め
        for ij in range(4, 21, 4):
            if self.enemy_pieces[ij] == 0:
                break
        else:
            return True

        return False

    def is_draw(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == 25

    def is_done(self):
        return self.is_lose() or self.is_draw()

    def next(self, action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_pieces, pieces)

    def legal_actions(self):
        actions = []
        for i in range(25):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions

    def is_first_player(self):
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        string = ''
        for i in range(25):
            if self.pieces[i] == 1:
                string += ox[0]
            elif self.enemy_pieces[i] == 1:
                string += ox[1]
            else:
                string += '-'
            if i % 5 == 4:
                string += '\n'
        return string


def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]


def alpha_beta(state, alpha, beta):
    # 負けは状態価値-1
    if state.is_lose():
        return -1

    # 引き分けは状態価値0
    if state.is_draw():
        return 0

    # 合法手の状態価値の計算
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha)
        if score > alpha:
            alpha = score

        # 現ノードのベストスコアが親ノードを超えたら探索終了
        if alpha >= beta:
            return alpha

    # 合法手の状態価値の最大値を返す
    return alpha


def alpha_beta_action(state):
    # 合法手の状態価値の計算
    best_action = 0
    alpha = -float('inf')
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha)
        if score > alpha:
            best_action = action
            alpha = score

    # 合法手の状態価値の最大値を持つ行動を返す
    return best_action


def playout(state):
    # 負けは状態価値-1
    if state.is_lose():
        return -1

    # 引き分けは状態価値0
    if state.is_draw():
        return 0

    # 次の状態の状態価値
    return -playout(state.next(random_action(state)))


def argmax(collection):
    return collection.index(max(collection))


def mcts_action(state):
    # モンテカルロ木探索のノード
    class node:
        # 初期化
        def __init__(self, state):
            self.state = state  # 状態
            self.w = 0  # 累計価値
            self.n = 0  # 試行回数
            self.child_nodes = None  # 子ノード群

        # 評価
        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                # 勝敗結果で価値を取得
                value = -1 if self.state.is_lose() else 0  # 負けは-1、引き分けは0

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            # 子ノードが存在しない時
            if not self.child_nodes:
                # プレイアウトで価値を取得
                value = playout(self.state)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                # 子ノードの展開
                if self.n == 10:
                    self.expand()
                return value

            # 子ノードが存在する時
            else:
                # UCB1が最大の子ノードの評価で価値を取得
                value = -self.next_child_node().evaluate()

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

        # 子ノードの展開
        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(node(self.state.next(action)))

        # UCB1が最大の子ノードを取得
        def next_child_node(self):
            # 試行回数nが0の子ノードを返す
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            # UCB1の計算
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(-child_node.w/child_node.n +
                                   2*(2*math.log(t)/child_node.n)**0.5)

            return self.child_nodes[argmax(ucb1_values)]

    # ルートノードの生成
    root_node = node(state)
    root_node.expand()

    for _ in range(100):
        root_node.evaluate()

    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]

def debug_action(state):
    legal_actions = state.legal_actions()
    while True:
        print('合法手：', end='')
        for la in legal_actions:
            print(la, end=' ')
        print()
        if state.is_first_player():
            inp = int(input(' o 先手の手 : '))
        else:
            inp = int(input(' x 後手の手 : '))
        if inp in legal_actions:
            break
    return inp


def main():
    state = State()
    while True:
        if state.is_done():
            break

        # action = debug_action(state)
        action = random_action(state)
        state = state.next(action)

        print(state)
        print('=====================')


if __name__ == '__main__':
    main()
