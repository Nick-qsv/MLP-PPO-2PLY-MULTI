from typing import List, Tuple

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

import torch

# Assume `tensor` is the input tensor of shape (n,)
# Assume `neural_network` is a PyTorch model that takes board tensors as input
# Assume `convert_to_board_tensor` is a function that converts moves into board tensors
# Assume `get_possible_moves` is a function that returns a list of possible moves for a dice roll


def weighted_state_value_average(
    tensor, get_possible_moves, convert_to_board_tensor, neural_network
):
    # 1. Extract the top 3 state values and their indices
    top_values, top_indices = torch.topk(tensor, k=3)

    # 2. Define the probabilities for each dice roll
    dice_rolls = [(roll, 1 / 36) for roll in range(1, 7)] + [
        (roll, 2 / 36) for roll in range(7, 22)
    ]

    # 3. Initialize the total weighted value and total probability
    weighted_sum = 0
    total_probability = 0

    # 4. Iterate through the top indices and their state values
    for index, value in zip(top_indices, top_values):
        # Iterate through each dice roll and its probability
        for dice_roll, probability in dice_rolls:
            # Get all possible moves for the dice roll
            possible_moves = get_possible_moves(dice_roll)

            # Convert moves into board tensors
            board_tensors = [convert_to_board_tensor(move) for move in possible_moves]

            # Evaluate the state values using the neural network
            if board_tensors:
                board_tensor_stack = torch.stack(board_tensors)
                state_values = neural_network(board_tensor_stack).squeeze()

                # Weight state values by their probabilities
                weighted_sum += (state_values * probability).sum().item()
                total_probability += probability * len(state_values)

    # 5. Return the weighted average
    return weighted_sum / total_probability if total_probability > 0 else 0


# Example usage
# tensor = torch.rand(100)  # Example tensor
# result = weighted_state_value_average(tensor, get_possible_moves, convert_to_board_tensor, neural_network)
