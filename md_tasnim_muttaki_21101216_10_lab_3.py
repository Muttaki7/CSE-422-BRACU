# -*- coding: utf-8 -*-
"""Md.Tasnim_Muttaki_21101216_10_lab-3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19xgh91DXH5NBNe6Hw1Bp0XbwsZyr8HRV
"""

#part-1
import random

class GameTree:
    def __init__(self, depth, player, max_depth, value=None):
        self.depth = depth
        self.player = player
        self.max_depth = max_depth
        self.value = value
        self.children = [ ]

    def build_tree(self):
        if self.depth < self.max_depth:
            for _ in range(2):
                next_player = 1 - self.player
                child = GameTree(self.depth + 1, next_player, self.max_depth)
                self.children.append(child)
                child.build_tree()

    def set_leaf_value(self, value):
        self.value = value

    def __str__(self):
        return ("Methods: build_tree, set_leaf_value")

def get_tree_leaves(node):
    if not node.children:
        return [node]
    leaves = [ ]
    for child in node.children:
        leaves.extend(get_tree_leaves(child))
    return leaves


def alpha_beta_pruning(node, alpha, beta, maximizing_player):
    if node.depth == node.max_depth or not node.children:
        return node.value

    if maximizing_player:
        max_eval = float('-inf')
        for child in node.children:
            eval = alpha_beta_pruning(child, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for child in node.children:
            eval = alpha_beta_pruning(child, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def mortal_kombat_game(start_player):
    max_depth = 5
    total_rounds = 3
    round_results = [ ]

    for _ in range(total_rounds):
        tree = GameTree(0, start_player, max_depth)
        tree.build_tree( )
        leaves = get_tree_leaves(tree)

        leaf_values = [random.choice([-1, 1]) for _ in range(len(leaves))]
        for i, leaf in enumerate(leaves):
            leaf.set_leaf_value(leaf_values[i])

        round_winner_value = alpha_beta_pruning(tree, float('-inf'), float('inf'), start_player == 0)
        round_winner = ("Scorpion") if round_winner_value == -1 else ("Sub-Zero")
        round_results.append(round_winner)

        start_player = 1 - start_player

    scorpion_wins = round_results.count("Scorpion")
    sub_zero_wins = round_results.count("Sub-Zero")
    game_winner = ("Scorpion") if scorpion_wins > sub_zero_wins else ("Sub-Zero")

    print("part-1: Mortal_Kombat_Game")
    print(f"Game Winner: {game_winner}")
    print(f"Total Rounds Played: {total_rounds}")
    for i, winner in enumerate(round_results):
        print(f"Winner of Round {i + 1}: {winner}")
    n = int(inputs[0].strip())
    t = int(inputs[1].strip())
    mortal_kombat_game(n)

#Part-2
def pacman_game(cost):
    leaf_values = [3, 6, 2, 3, 7, 1, 2, 0]

    def minimax(depth, index, maximizing_player, leaf_values, alpha, beta):
        if depth == 3:
            return leaf_values[index]

        if maximizing_player:
            max_eval = float('-inf')
            for i in range(2):
                eval = minimax(depth + 1, index * 2 + i, False, leaf_values, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(2):
                eval = minimax(depth + 1, index * 2 + i, True, leaf_values, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    final_value = minimax(0, 0, True, leaf_values, float('-inf'), float('inf'))

    left_with_dark_magic = leaf_values[1] - cost
    right_with_dark_magic = leaf_values[4] - cost
    best_with_dark_magic = max(left_with_dark_magic, right_with_dark_magic)

    print("\nPart-2: Pacman Game")
    if best_with_dark_magic > final_value:
        if left_with_dark_magic > right_with_dark_magic:
            print(f"The new minimax value is {best_with_dark_magic}. Pacman goes left and uses dark magic.")
        else:
            print(f"The new minimax value is {best_with_dark_magic}. Pacman goes right and uses dark magic.")
    else:
        print(f"The minimax value is {final_value}. Pacman does not use dark magic.")
    n = int(inputs[0].strip())
    t = int(inputs[1].strip())
    pacman_game(t)