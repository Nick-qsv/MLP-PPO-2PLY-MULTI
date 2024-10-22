from typing import Tuple

from src.backgammon.board import ImmutableBoard
from src.backgammon.moves.conditions import (
    check_if_blot,
    is_valid_entry_at_index,
    valid_move,
    compute_board_state,
)
from src.backgammon.types import Player, Position, SubMove, NUMBER_OF_POINTS, BoardState


def get_moves_with_one_die(
    board: ImmutableBoard, die_value: int, player: Player
) -> list:
    """
    Generates all possible sub-moves for a player given a single die value based on the current board state.

    Args:
        board (ImmutableBoard): The current state of the board.
        die_value (int): The value of the die rolled.
        player (Player): The player for whom to generate moves.

    Returns:
        list: A list of possible SubMove instances.
    """
    # Determine the current state of the board for the player
    current_board_state = compute_board_state(board, player)

    if current_board_state == BoardState.NORMAL:
        return get_moves_normal(board, die_value, player)
    elif current_board_state == BoardState.ON_BAR:
        return get_moves_bar(board, die_value, player)
    elif current_board_state == BoardState.BEAR_OFF:
        return get_moves_bear_off(board, die_value, player)
    else:
        return []  # No moves available (e.g., GAME_OVER)


def get_moves_normal(
    board: ImmutableBoard, die_value: int, player: Player
) -> Tuple[SubMove, ...]:
    """
    Retrieves all valid normal sub-moves for the player based on the die value.

    Args:
        board (ImmutableBoard): The current state of the board.
        die_value (int): The value of the die rolled.
        player (Player): The player for whom to generate moves.

    Returns:
        Tuple[SubMove, ...]: A tuple of valid SubMove instances.
    """
    moves = []
    player_idx = player.value  # 0 for PLAYER1, 1 for PLAYER2
    direction = 1 if player == Player.PLAYER1 else -1

    # Select the appropriate positions tuple based on the player
    player_positions = (
        board.positions_0 if player == Player.PLAYER1 else board.positions_1
    )

    for idx in range(NUMBER_OF_POINTS):
        own_checkers = player_positions[idx]
        if own_checkers > 0:
            # Calculate the destination index based on direction and die value
            destination_idx = idx + die_value * direction

            # Only consider moves within the board (exclude bear-off)
            if 0 <= destination_idx < NUMBER_OF_POINTS:
                if valid_move(destination_idx, player, board):
                    # Determine if the destination has a blot to hit
                    hits_blot = check_if_blot(destination_idx, player, board)

                    # Create SubMove with Position enums
                    start_position = Position(idx)
                    end_position = Position(destination_idx)

                    sub_move = SubMove(
                        start=start_position, end=end_position, hits_blot=hits_blot
                    )
                    moves.append(sub_move)

    return tuple(moves)


def get_moves_bar(
    board: ImmutableBoard, die_value: int, player: Player
) -> Tuple[SubMove, ...]:
    """
    Generates all valid sub-moves for a player who has checkers on the bar based on the die value.

    Args:
        board (ImmutableBoard): The current state of the board.
        die_value (int): The value of the die rolled.
        player (Player): The player for whom to generate moves.

    Returns:
        Tuple[SubMove, ...]: A tuple of valid SubMove instances for entering from the bar.
    """
    moves = []

    # Check if the player has any checkers on the bar
    player_bar = board.bar[player.value]
    if player_bar == 0:
        return tuple(moves)  # No checkers on the bar, no moves to generate

    # Determine the destination index based on player perspective
    if player == Player.PLAYER1:
        dest_idx = die_value - 1  # PLAYER1 enters from points 0 to 5
        valid_home_indices = range(0, 6)  # 0 to 5 inclusive
    else:
        dest_idx = 24 - die_value  # PLAYER2 enters from points 23 to 18
        valid_home_indices = range(18, 24)  # 18 to 23 inclusive

    # Check if the destination index is within the home board
    if dest_idx in valid_home_indices:
        # Check if the entry point is valid (not blocked by two or more opponent checkers)
        if is_valid_entry_at_index(index=dest_idx, player=player, board=board):
            start_position = Position.BAR  # Starting from the bar
            end_position = Position(dest_idx)  # Destination within home board

            # Determine if the destination has a blot to hit
            hits_blot = check_if_blot(index=dest_idx, player=player, board=board)

            sub_move = SubMove(
                start=start_position, end=end_position, hits_blot=hits_blot
            )
            moves.append(sub_move)

    return tuple(moves)


def get_moves_bear_off(
    board: ImmutableBoard, die_value: int, player: Player
) -> Tuple[SubMove, ...]:
    """
    Generates all valid bear-off sub-moves for the player based on the die value.

    This function first finds all normal moves within the player's home board.
    It then identifies the farthest checker from the exit point and handles
    bear-off moves accordingly, ensuring no duplicate moves are added.

    Args:
        board (ImmutableBoard): The current state of the board.
        die_value (int): The value of the die rolled.
        player (Player): The player for whom to generate bear-off moves.

    Returns:
        Tuple[SubMove, ...]: A tuple of valid SubMove instances for bearing off.
    """
    move_set = []
    player_idx = player.value  # 0 for PLAYER1, 1 for PLAYER2

    # Define home board indices and movement direction based on player
    if player == Player.PLAYER1:
        home_indexes = list(range(18, 24))  # PLAYER1's home board: points 18-23
        direction = 1  # Moving towards higher indices
        last_checker_idx = 18  # Initialize with the first home point
    else:
        home_indexes = list(range(0, 6))  # PLAYER2's home board: points 0-5
        direction = -1  # Moving towards lower indices
        last_checker_idx = 5  # Initialize with the last home point

    # 1. Normal moves within the home board
    for idx in home_indexes:
        own_checkers = (
            board.positions_0[idx]
            if player == Player.PLAYER1
            else board.positions_1[idx]
        )
        if own_checkers > 0:
            destination_idx = idx + die_value * direction
            # Ensure destination is within the board for normal moves (exclude bear-off)
            if 0 <= destination_idx < NUMBER_OF_POINTS:
                if valid_move(
                    destination_idx=destination_idx, player=player, board=board
                ):
                    # Determine if the destination has a blot to hit
                    hits_blot = check_if_blot(
                        index=destination_idx, player=player, board=board
                    )

                    # Create SubMove with Position enums
                    start_position = Position(idx)
                    end_position = Position(destination_idx)

                    sub_move = SubMove(
                        start=start_position, end=end_position, hits_blot=hits_blot
                    )
                    move_set.append(sub_move)

    # 2. Find the index of the farthest checker from the exit point (last checker)
    if player == Player.PLAYER1:
        # Iterate from lowest to highest index in home board to find the first occupied point
        for idx in home_indexes:
            if board.positions_0[idx] > 0:
                last_checker_idx = idx
                break
    else:
        # Iterate from highest to lowest index in home board to find the first occupied point
        for idx in reversed(home_indexes):
            if board.positions_1[idx] > 0:
                last_checker_idx = idx
                break

    # 3. Handle bear-off moves
    if player == Player.PLAYER1:
        # Standard bear-off move: Check if the farthest checker can bear off
        if last_checker_idx + die_value * direction >= NUMBER_OF_POINTS:
            sub_move = SubMove(
                start=Position(last_checker_idx),
                end=Position.BEAR_OFF,
                hits_blot=False,  # Bearing off does not hit blots
            )
            move_set.append(sub_move)

        # Potential special bear-off move
        potential_start_index = NUMBER_OF_POINTS - die_value
        if potential_start_index != last_checker_idx:
            # Ensure potential_start_index is within home board
            if potential_start_index in home_indexes:
                if board.positions_0[potential_start_index] > 0:
                    sub_move = SubMove(
                        start=Position(potential_start_index),
                        end=Position.BEAR_OFF,
                        hits_blot=False,
                    )
                    move_set.append(sub_move)
    else:
        # Standard bear-off move: Check if the farthest checker can bear off
        if last_checker_idx + die_value * direction < 0:
            sub_move = SubMove(
                start=Position(last_checker_idx),
                end=Position.BEAR_OFF,
                hits_blot=False,  # Bearing off does not hit blots
            )
            move_set.append(sub_move)

        # Potential special bear-off move
        potential_start_index = die_value - 1
        if potential_start_index != last_checker_idx:
            # Ensure potential_start_index is within home board
            if potential_start_index in home_indexes:
                if board.positions_1[potential_start_index] > 0:
                    sub_move = SubMove(
                        start=Position(potential_start_index),
                        end=Position.BEAR_OFF,
                        hits_blot=False,
                    )
                    move_set.append(sub_move)

    return tuple(move_set)
