import random

def tit_for_tat(history):
    if len(history) == 0:
        return 'C'
    else:
        return history[-1]
    
def tit_for_two_tats(history):
    if len(history) < 2:
        return 'C'
    elif history[-1] == 'D' and history[-2] == 'D':
        return 'D'
    else:
        return 'C'

def play_round(player1_strategy, player2_strategy, player1_history, player2_history):
    p1_move = player1_strategy(player2_history)
    p2_move = player2_strategy(player1_history)

    if p1_move == 'C' and p2_move == 'C':
        return 3, 3
    elif p1_move == 'C' and p2_move == 'D':
        return 0, 5
    elif p1_move == 'D' and p2_move == 'C':
        return 5, 0
    else:
        return 1, 1

def play_game(player1_strategy, player2_strategy, rounds):
    player1_history = []
    player2_history = []
    player1_score = 0
    player2_score = 0

    for _ in range(rounds):
        p1_score, p2_score = play_round(player1_strategy, player2_strategy, player1_history, player2_history)
        player1_score += p1_score
        player2_score += p2_score

        p1_move, p2_move = player1_strategy(player2_history), player2_strategy(player1_history)  # Defining p1_move and p2_move here

        player1_history.append(p1_move)
        player2_history.append(p2_move)

    return player1_score, player2_score

# Example usage:
strategies = [tit_for_tat, tit_for_two_tats]

for strategy1 in strategies:
    for strategy2 in strategies:
        score1, score2 = play_game(strategy1, strategy2, 100)
        print(f"{strategy1.__name__} vs {strategy2.__name__}: Player 1 score: {score1}, Player 2 score: {score2}")
