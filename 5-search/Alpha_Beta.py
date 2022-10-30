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


def alpha_beta(state, alpha, beta):
    if state.is_lose():
        return -1

    if state.is_draw():
        return 0

    for action in state.legal_action():
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
            action = mini_max_plus(state, 3)
        else:
            action = Mini_Max.random_action(state)

        state = state.next(action)

        print(state)
        print()


if __name__ == '__main__':
    main()
