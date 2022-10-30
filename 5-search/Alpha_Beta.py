import Mini_Max


def mini_max_plus(state, limit):
    if state.is_lose():
        return -1

    if state.is_draw():
        return 0

    best_score = -float('inf')
    for action in state.legal_actions():
        score = -mini_max_plus(state.next(action), -best_score)
        if score > best_score:
            best_score = score

        if best_score >= limit:
            return best_score

    return best_score


def alpha_beta_action(state):
    best_action = 0
    alpha = -float('inf')
    string = ['', '']
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha)
        if score > alpha:
            best_action = action
            alpha = score

        string[0] = '{}{:3d},'.format(string[0], action)
        string[1] = '{}{:3d},'.format(string[1], score)
    print('action : ', string[0], '\nscore : ', string[1], '\n')
    return best_action


def alpha_beta(state, alpha, beta):
    if state.is_lose():
        return -1

    if state.is_draw():
        return 0

    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha)
        if score > alpha:
            alpha = score

        if alpha >= beta:
            return alpha

    return alpha


def main():
    state = Mini_Max.State()
    while True:
        if state.is_done():
            break

        if state.is_first_player():
            action = alpha_beta_action(state)
        else:
            action = Mini_Max.mini_max_action(state)

        state = state.next(action)

        print(state)
        print()


if __name__ == '__main__':
    main()
