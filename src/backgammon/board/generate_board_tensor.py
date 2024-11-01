import numpy as np
import torch
from typing import List
from src.backgammon.board.immutable_board import ImmutableBoard
from src.backgammon.types import FullMove, Player, Position


def generate_all_board_features(
    board: ImmutableBoard,
    current_player: Player,
    legal_moves: List[FullMove],
) -> torch.Tensor:
    """
    Generates a tensor of all possible board features based on legal moves.
    """
    N = len(legal_moves)
    features = torch.zeros(N, 198, dtype=torch.float32)

    # Convert board positions to NumPy arrays for efficient manipulation
    positions_0_initial = np.array(board.positions_0, dtype=np.int8)
    positions_1_initial = np.array(board.positions_1, dtype=np.int8)
    bar_initial = np.array(board.bar, dtype=np.int8)
    borne_off_initial = np.array(board.borne_off, dtype=np.int8)

    # Preallocate arrays to avoid creating new ones in each iteration
    p0 = positions_0_initial.copy()
    p1 = positions_1_initial.copy()
    b = bar_initial.copy()
    bo = borne_off_initial.copy()

    for i, move in enumerate(legal_moves):
        # Reset the arrays to the initial board state
        p0[:] = positions_0_initial
        p1[:] = positions_1_initial
        b[:] = bar_initial
        bo[:] = borne_off_initial

        # Apply the move to the board arrays
        apply_full_move(p0, p1, b, bo, current_player, move)

        # Compute features from the updated board arrays
        features[i, :] = compute_features(p0, p1, b, bo, current_player)

    return features


def apply_full_move(p0, p1, b, bo, player, full_move):
    """
    Applies a full move to the board arrays.
    """
    player_idx = player.value
    opponent_idx = 1 - player_idx

    for sub_move in full_move.sub_move_commands:
        apply_sub_move(p0, p1, b, bo, player_idx, opponent_idx, sub_move)


def apply_sub_move(p0, p1, b, bo, player_idx, opponent_idx, sub_move):
    start = sub_move.start
    end = sub_move.end
    hits_blot = sub_move.hits_blot

    # Remove checker from start
    if start == Position.BAR:
        if b[player_idx] <= 0:
            raise ValueError(
                f"No checker to remove from bar for Player {player_idx + 1}."
            )
        b[player_idx] -= 1
    else:
        player_positions = p0 if player_idx == 0 else p1
        if player_positions[start.value] <= 0:
            raise ValueError(
                f"No checker to remove at point {start.name} for Player {player_idx + 1}."
            )
        player_positions[start.value] -= 1

    # Handle hitting a blot
    if hits_blot:
        opponent_positions = p0 if opponent_idx == 0 else p1
        if opponent_positions[end.value] == 1:
            opponent_positions[end.value] -= 1
            b[opponent_idx] += 1
        else:
            raise ValueError(
                f"No blot to hit at point {end.name} for Player {player_idx + 1}."
            )

    # Add checker to end
    if end == Position.BEAR_OFF:
        bo[player_idx] += 1
    else:
        player_positions = p0 if player_idx == 0 else p1
        player_positions[end.value] += 1


def compute_features(p0, p1, b, bo, current_player):
    """
    Computes the 198-dimensional feature vector for the board state.
    """
    features = torch.zeros(198, dtype=torch.float32)
    feature_index = 0

    for player_idx, player_positions in enumerate([p0, p1]):
        for point_idx in range(24):
            checkers = player_positions[point_idx]
            features_slice = torch.zeros(4, dtype=torch.float32)

            if checkers == 1:
                features_slice[0] = 1.0
            elif checkers == 2:
                features_slice[0:2] = 1.0
            elif checkers >= 3:
                features_slice[0:3] = 1.0
                features_slice[3] = (checkers - 3.0) / 2.0

            features[feature_index : feature_index + 4] = features_slice
            feature_index += 4

        # Bar checkers
        bar_checkers = b[player_idx]
        features[feature_index] = bar_checkers / 2.0
        feature_index += 1

        # Borne off checkers
        borne_off_checkers = bo[player_idx]
        features[feature_index] = borne_off_checkers / 15.0
        feature_index += 1

    # Add current player indicator
    features[feature_index] = 1.0 if current_player == Player.PLAYER1 else 0.0
    features[feature_index + 1] = 1.0 if current_player == Player.PLAYER2 else 0.0
    feature_index += 2

    assert (
        feature_index == 198
    ), f"Feature vector length is {feature_index}, expected 198"

    return features
