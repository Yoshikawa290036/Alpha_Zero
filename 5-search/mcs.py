import Mini_Max
import Alpha_Beta
import numpy as np

EP_GAME_COUNT = 100


def playout(state):
    if state.is_lose():
        return -1

    if state.is_draw():
        return 0

    return -playout(state.next(Mini_Max.random_action(state)))


def argmax(collection, key=None):
    return collection.index(max(collection))


def mcs_action(state):
    legal_actions = state.legal_actions()
    values = [0] * len(legal_actions)

    for i, action in enumerate(legal_actions):
        for _ in range(10):
            values[i] += -playout(state.next(action))

    return legal_actions[argmax(values)]


def first_player_point(ended_state):
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5


def play(next_actions):
    state = Mini_Max.State()

    while True:
        if state.is_done():
            break

        next_action = next_actions[0] if state.is_first_player(
        ) else next_actions[1]
        action = next_action(state)

        state = state.next(action)

    return first_player_point(state)


def evaluate_algorithm_of(label, next_actions):
    total_point = 0
    for i in range(EP_GAME_COUNT):
        if not i % 2:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        print('\rEvaluate {}/{}'.format(i+1, EP_GAME_COUNT), end='')
    print()

    average_point = total_point / EP_GAME_COUNT
    print(label.format(average_point))


def main():
    next_actions = (mcs_action, Mini_Max.random_action)
    evaluate_algorithm_of('VS Random {:.3f}', next_actions)

    next_actions = (mcs_action, Alpha_Beta.alpha_beta_action)
    evaluate_algorithm_of('VS AlphaBeta {:.3f}', next_actions)


if __name__ == '__main__':
    main()
