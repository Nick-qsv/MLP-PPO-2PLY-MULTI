from typing import List, Tuple
import random
import torch

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


def select_best_move(
    initial_board_state,
    current_player,
    neural_network,
    alpha=1.0,
    beta=0.5,
):
    """
    Selects the best move based on an extended weighted evaluation model.

    Parameters:
    - initial_board_state: The current board state (ImmutableBoard).
    - current_player: The player making the move (Player).
    - neural_network: An instance of BackgammonPolicyNetwork.
    - alpha: Weight for the move's state value (default: 1.0).
    - beta: Weight for the opponent's expected response value (default: 0.5).

    Returns:
    - best_move: The move with the highest score.
    - best_score: The highest score achieved.
    """

    # Get all possible moves for the current player and current board state
    dice_roll = initial_board_state.current_dice_roll  # Assuming this is available
    possible_moves = get_all_possible_moves(
        current_player, initial_board_state, dice_roll
    )

    # If no possible moves, return None
    if not possible_moves:
        return None, None

    # Initialize variables to keep track of the best move and score
    best_move = None
    best_score = float("-inf")

    # Iterate over each possible move
    for move in possible_moves:
        # Apply the move to get the new board state
        new_board_state = execute_full_move_on_board_copy(initial_board_state, move)

        # Evaluate the state value S_m of the move using the neural network
        feature_vector = new_board_state.get_board_features(current_player)
        S_m = neural_network.forward(feature_vector.unsqueeze(0)).item()

        # Determine opponent's possible responses O_m
        opponent = current_player.opponent()
        W_O_m = compute_weighted_opponent_response(
            new_board_state,
            opponent,
            neural_network,
        )

        # Compute the score for the move
        score = alpha * S_m - beta * W_O_m

        # Update the best move if this move has a higher score
        if score > best_score:
            best_score = score
            best_move = move

    return best_move, best_score


def compute_weighted_opponent_response(
    board_state,
    opponent_player,
    neural_network,
):
    """
    Computes the weighted average of the opponent's responses.

    Parameters:
    - board_state: The board state after the player's move.
    - opponent_player: The opponent player.
    - neural_network: An instance of BackgammonPolicyNetwork.

    Returns:
    - W_O_m: The weighted average of the opponent's possible responses.
    """
    total_weighted_value = 0.0
    total_probability = 0.0

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
            state_values = neural_network.forward(board_tensors).squeeze()

            # Accumulate weighted values
            total_weighted_value += (state_values * probability).sum().item()
            total_probability += probability * len(state_values)

    # Compute the weighted average
    W_O_m = total_weighted_value / total_probability if total_probability > 0 else 0.0
    return W_O_m
