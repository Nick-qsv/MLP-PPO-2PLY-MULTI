from backgammon.types.moves import Player, FullMove, Position
from backgammon.board.immutable_board import ImmutableBoard
import torch
from typing import List


def generate_all_board_features(
    board: ImmutableBoard,
    current_player: Player,
    legal_moves: List[FullMove],
) -> torch.Tensor:
    """
    Generates a tensor of all possible board features based on legal moves.
    """
    N = len(legal_moves)
    features = torch.zeros(N, 198, dtype=torch.float32, device=board.device)
    for i, full_move in enumerate(legal_moves):
        # Apply the full move to a copy of the board
        new_board = execute_full_move_on_board_copy(board, full_move)
        # Get the feature vector of the new board state
        feature_vector = new_board.get_board_features(current_player)
        # Store the feature vector in the features tensor
        features[i] = feature_vector
    return features


def execute_full_move_on_board_copy(
    board: ImmutableBoard, full_move: FullMove
) -> ImmutableBoard:
    player = full_move.player
    # opponent = Player(1 - player)

    if player == Player.PLAYER1:
        positions_player = board.positions_0
        positions_opponent = board.positions_1
        bar_player = board.bar[Player.PLAYER1]
        bar_opponent = board.bar[Player.PLAYER2]
        borne_off_player = board.borne_off[Player.PLAYER1]
        borne_off_opponent = board.borne_off[Player.PLAYER2]
    else:
        positions_player = board.positions_1
        positions_opponent = board.positions_0
        bar_player = board.bar[Player.PLAYER2]
        bar_opponent = board.bar[Player.PLAYER1]
        borne_off_player = board.borne_off[Player.PLAYER2]
        borne_off_opponent = board.borne_off[Player.PLAYER1]

    positions_player_list = list(positions_player)
    positions_opponent_list = list(positions_opponent)

    for submove in full_move.sub_move_commands:
        start = submove.start
        end = submove.end
        hits_blot = submove.hits_blot

        # Update start position
        if start == Position.BAR:
            bar_player -= 1
        else:
            positions_player_list[start] -= 1

        # Handle hitting opponent's blot
        if hits_blot:
            positions_opponent_list[end] -= 1
            bar_opponent += 1

        # Update end position
        if end == Position.BEAR_OFF:
            borne_off_player += 1
        else:
            positions_player_list[end] += 1

    # Prepare new positions and counts
    if player == Player.PLAYER1:
        new_positions_0 = tuple(positions_player_list)
        new_positions_1 = tuple(positions_opponent_list)
        new_bar = (bar_player, bar_opponent)
        new_borne_off = (borne_off_player, borne_off_opponent)
    else:
        new_positions_0 = tuple(positions_opponent_list)
        new_positions_1 = tuple(positions_player_list)
        new_bar = (bar_opponent, bar_player)
        new_borne_off = (borne_off_opponent, borne_off_player)

    return ImmutableBoard(
        positions_0=new_positions_0,
        positions_1=new_positions_1,
        bar=new_bar,
        borne_off=new_borne_off,
        device=board.device,
    )


def execute_full_move_on_board_copy_old(
    board: ImmutableBoard, full_move: FullMove
) -> ImmutableBoard:
    """
    Executes a full move on a copy of the board by applying each sub-move sequentially.

    Args:
        board (ImmutableBoard): The current immutable board state.
        full_move (FullMove): The full move consisting of sub-moves to execute.

    Returns:
        ImmutableBoard: A new board state after executing the full move.
    """
    new_board = board
    for sub_move in full_move.sub_move_commands:
        new_board = new_board.move_checker(full_move.player, sub_move)
    return new_board


def check_game_over(board: ImmutableBoard, current_player: Player) -> bool:
    """
    Checks if the current player has won the game by bearing off all 15 checkers.
    """
    return board.borne_off[current_player.value] >= 15


def check_for_gammon(board: ImmutableBoard, player: Player) -> bool:
    """
    Checks if the player has won a gammon against the opponent.
    A gammon occurs when the opponent has not borne off any checkers.
    """
    opponent = get_opponent(player)
    opponent_borne_off = board.borne_off[opponent.value]
    return opponent_borne_off == 0


def check_for_backgammon(board: ImmutableBoard, player: Player) -> bool:
    """
    Checks if the player has won a backgammon against the opponent.
    A backgammon occurs when the opponent has not borne off any checkers and
    has checkers on the bar or in the player's home board.
    """
    opponent = get_opponent(player)
    opponent_idx = opponent.value

    # Check if the opponent has borne off any checkers
    opponent_borne_off = board.borne_off[opponent_idx]
    if opponent_borne_off > 0:
        return False  # Opponent has borne off at least one checker

    # Define the home board range for the current player
    if player == Player.PLAYER1:
        home_board_indices = range(18, 24)  # Points 18-23
    else:
        home_board_indices = range(0, 6)  # Points 0-5

    # Get opponent's positions
    opponent_positions = board.positions_1 if opponent_idx == 1 else board.positions_0

    # Check if opponent has any checkers in the player's home board
    for idx in home_board_indices:
        if opponent_positions[idx] > 0:
            return True  # Opponent has a checker in player's home board

    # Check if opponent has any checkers on the bar
    opponent_bar = board.bar[opponent_idx]
    if opponent_bar > 0:
        return True  # Opponent has checkers on the bar

    return False  # No checkers in home board or on the bar


def get_opponent(player: Player) -> Player:
    return Player.PLAYER2 if player == Player.PLAYER1 else Player.PLAYER1
