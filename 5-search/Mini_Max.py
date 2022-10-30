import random


class State:
    def __init__(self, piece=None, enemy_piece=None):
        self.pieces = piece if piece != None else [0]*9
        self.enemy_piece = enemy_piece if enemy_piece != None else [0]*9

    def piece_count(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count += 1
        return count

    def is_lose(self):
        def is_comp(x, y, dx, dy):
            for k in range(3):
                if y < 0 or 2 < y or x < 0 or 2 < x or \
                        self.enemy_piece[x+y*3] == 0:
                    return False
                x, y = x+dx, y+dy
            return True

        if is_comp(0, 0, 1, 1) or is_comp(0, 2, 1, -1):
            return True
        for i in range(3):
            if is_comp(0, i, 1, 0) or is_comp(i, 0, 0, 1):
                return True
        return False

    def is_draw(self):
        return self.piece_count(self.pieces)+self.piece_count(self.enemy_piece) == 9

    def is_done(self):
        return self.is_lose() or self.is_draw()

    def next(self, action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_piece, pieces)

    def legal_actions(self):
        actions = []
        for i in range(9):
            if self.pieces[i] == 0 and self.enemy_piece[i] == 0:
                actions.append(i)
        return actions

    def is_first_player(self):
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_piece)

    def __str__(self):
        ox = ('o', 'x')if self.is_first_player() else ('x', 'o')
        string = ''
        for i in range(9):
            if self.pieces[i] == 1:
                string += ox[0]
            elif self.enemy_piece[i] == 1:
                string += ox[1]
            else:
                string += '-'
            if i % 3 == 2:
                string += '\n'
        return string


def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions)-1)]


def mini_max(state):
    if state.is_lose():
        return -1
    if state.is_draw():
        return 0
    best_score = -float('inf')
    for action in state.legal_actions():
        score = -mini_max(state.next(action))
        if score > best_score:
            best_score = score
    return best_score


def mini_max_action(state):
    best_action = 0
    best_score = -float('inf')
    string = ['', '']
    for action in state.legal_actions():
        score = -mini_max(state.next(action))
        if score > best_score:
            best_action = action
            best_score = score

        string[0] = '{}{:3d},'.format(string[0], action)
        string[1] = '{}{:3d},'.format(string[1], score)
    print('action : ', string[0], '\nscore : ', string[1], '\n')
    return best_action


def main():
    state = State()
    while True:
        if state.is_done():
            break

        if state.is_first_player():
            action = mini_max_action(state)
        else:
            action = random_action(state)

        state = state.next(action)

        print(state)
        print()


if __name__ == '__main__':
    main()
