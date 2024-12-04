from typing import List, Tuple
import random
import torch
from src.environments import generate_all_board_features
from src.backgammon.board.immutable_board import ImmutableBoard
from src.backgammon.types import Player
from src.backgammon import get_all_possible_moves

# Precomputed dice rolls and probabilities
DICE_ROLLS = [
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 6],
    [2, 2],
    [2, 3],
    [2, 4],
    [2, 5],
    [2, 6],
    [3, 3],
    [3, 4],
    [3, 5],
    [3, 6],
    [4, 4],
    [4, 5],
    [4, 6],
    [5, 5],
    [5, 6],
    [6, 6],
]
COUNTS = [1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1]
TOTAL_OUTCOMES = 36
PROBABILITIES = [count / TOTAL_OUTCOMES for count in COUNTS]


# Assume `tensor` is the input tensor of shape (n,)
# Assume `neural_network` is a PyTorch model that takes board tensors as input
# Assume `convert_to_board_tensor` is a function that converts moves into board tensors
# Assume `get_possible_moves` is a function that returns a list of possible moves for a dice roll


def compute_scores_for_boards(
    boards: list,  # List of 4 ImmutableBoard objects
    state_values: list,  # List of 4 state values (S_m)
    env,
    alpha=1.0,
    beta=0.9,
):
    """
    Computes scores for a list of boards and corresponding state values.

    Parameters:
    - boards: A list of exactly 4 ImmutableBoard objects.
    - state_values: A list of exactly 4 state values (S_m), one for each board.
    - env: The environment containing relevant evaluation tools.
    - alpha: Weight for the move's state value (default: 1.0).
    - beta: Weight for the opponent's expected response value (default: 0.5).

    Returns:
    - scores: A list of computed scores for each board.
    """

    # Validate inputs
    if len(boards) != 4:
        raise ValueError("The 'boards' input must contain exactly 4 elements.")
    if len(state_values) != 4:
        raise ValueError("The 'state_values' input must contain exactly 4 elements.")

    scores = []

    # Loop through each board and corresponding state value
    for board, S_m in zip(boards, state_values):
        # Compute the weighted opponent response for the current board
        W_O_m = compute_weighted_opponent_response(
            board, board.current_player.opponent(), env
        )

        # Compute the score for the board
        score = alpha * S_m - beta * W_O_m

        # Add the computed score to the list
        scores.append(score)

    return scores


def compute_weighted_opponent_response(
    board_state: ImmutableBoard,
    opponent_player: Player,
    env,
):
    """
    Computes the weighted average of the opponent's responses,
    considering only the top 5 moves per dice roll.

    Parameters:
    - board_state: The board state after the player's move.
    - opponent_player: The opponent player.
    - env: An environment object containing the policy network.

    Returns:
    - W_O_m: The weighted average of the opponent's possible responses.
    """
    total_weighted_average = 0.0

    # Iterate over all possible dice rolls and their probabilities
    for dice_roll, probability in zip(DICE_ROLLS, PROBABILITIES):
        # Get all possible opponent moves for this dice roll
        opponent_moves = get_all_possible_moves(opponent_player, board_state, dice_roll)

        # Limit moves for specific dice rolls
        if dice_roll in ([1, 1], [2, 2], [3, 3]):
            if len(opponent_moves) > 50:
                opponent_moves = random.sample(opponent_moves, 50)

        if opponent_moves:
            # Generate board tensors for opponent's possible moves
            board_tensors = generate_all_board_features(
                board_state,
                opponent_player,
                opponent_moves,
            )

            # Evaluate state values of opponent's moves
            state_values = env.policy_network.forward(board_tensors).squeeze()

            # Sort state values in descending order
            sorted_state_values, _ = torch.sort(state_values, descending=True)

            # Select the top 5 state values (or fewer)
            top_state_values = sorted_state_values[:5]

            # Compute the average of the top state values
            average_top_value = top_state_values.mean().item()

            # Multiply by the roll's probability and accumulate
            weighted_average = average_top_value * probability
            total_weighted_average += weighted_average

    # The final weighted average
    W_O_m = total_weighted_average
    return W_O_m
