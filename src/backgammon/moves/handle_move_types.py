from typing import Tuple, List, Set
from src.backgammon.board.immutable_board import ImmutableBoard
from src.backgammon.types import SubMove, FullMove, Player
from src.backgammon.moves.get_moves_one_die import get_moves_with_one_die


def handle_non_doubles(
    board: ImmutableBoard,
    roll: List[int],
    full_moves: List[FullMove],
    unique_boards: Set[ImmutableBoard],
    player: Player,
    reverse: bool = False,
):
    """
    Handles move generation for non-double die rolls.

    This function generates all possible move sequences for a player based on two distinct die values.
    According to Backgammon rules, the player must make as many moves as possible using the dice rolled.
    If both dice can be used, the player must use both. If only one die can be used, the player must use
    the higher die if possible.

    Args:
        board (ImmutableBoard): The current state of the board.
        roll (List[int]): A list containing two die values.
        full_moves (List[FullMove]): A list to store all unique full move sequences.
        unique_boards (Set[ImmutableBoard]): A set to keep track of already processed board states.
        player (Player): The player for whom to generate moves.
        reverse (bool, optional): If True, reverses the order of die application. Defaults to False.
    """
    # Determine the order of dice based on the reverse flag
    dice_order = [roll[1], roll[0]] if reverse else [roll[0], roll[1]]

    # Generate all possible initial moves using the first die
    first_die_moves = get_moves_with_one_die(
        board=board, die_value=dice_order[0], player=player
    )

    # Flag to check if any two-move sequences are generated
    two_move_sequences_generated = False

    # Attempt to generate all possible two-move sequences
    for initial_move in first_die_moves:
        # Apply the initial move to get the resulting board state
        resulting_board = board.move_checker(player=player, sub_move=initial_move)

        # Generate all possible second moves based on the second die
        second_die_moves = get_moves_with_one_die(
            board=resulting_board, die_value=dice_order[1], player=player
        )

        if second_die_moves:
            two_move_sequences_generated = True  # At least one two-move sequence exists
            for follow_up_move in second_die_moves:
                # Apply the follow-up move to get the new board state
                board_after_second_move = resulting_board.move_checker(
                    player=player, sub_move=follow_up_move
                )

                # Record the full move sequence (initial move + follow-up move)
                moves = (initial_move, follow_up_move)
                add_unique_board(
                    board=board_after_second_move,
                    moves=moves,
                    full_moves=full_moves,
                    unique_boards=unique_boards,
                    player=player,
                )

    # If no two-move sequences were generated, add single-move sequences
    if not two_move_sequences_generated:
        for initial_move in first_die_moves:
            resulting_board = board.move_checker(player=player, sub_move=initial_move)
            moves = (initial_move,)
            add_unique_board(
                board=resulting_board,
                moves=moves,
                full_moves=full_moves,
                unique_boards=unique_boards,
                player=player,
            )


def handle_doubles(
    board: ImmutableBoard,
    die_value: int,
    full_moves: List[FullMove],
    unique_boards: Set[ImmutableBoard],
    player: Player,
):
    """
    Handles move generation for double die rolls.

    This function generates all possible move sequences for a player based on a double die roll.
    Since doubles allow the player to make four moves instead of two, the function generates move
    sequences up to four sub-moves, ensuring that each sequence is unique.

    Args:
        board (ImmutableBoard): The current state of the board.
        die_value (int): The value of the die rolled (both die values are the same).
        full_moves (List[FullMove]): A list to store all unique full move sequences.
        unique_boards (Set[ImmutableBoard]): A set to keep track of already processed board states.
        player (Player): The player for whom to generate moves.
    """
    # Generate all possible first moves using the die value
    single_die_moves = get_moves_with_one_die(board, die_value, player)
    # Flag to check if a full move of length 4 is possible
    full_move_of_length_4_possible = False

    for first_move in single_die_moves:
        # Apply the first move to get the resulting board state
        first_board = board.move_checker(player=player, sub_move=first_move)

        # Generate all possible second moves using the same die value
        second_die_moves = get_moves_with_one_die(first_board, die_value, player)

        if (
            not second_die_moves
            and len(single_die_moves) == 1
            and not full_move_of_length_4_possible
        ):
            # If no second moves are possible and only one first move exists, record it
            add_unique_board(
                board=first_board,
                moves=(first_move,),
                full_moves=full_moves,
                unique_boards=unique_boards,
                player=player,
            )

        for second_move in second_die_moves:
            # Apply the second move to get the new board state
            second_board = second_board = first_board.move_checker(
                player=player, sub_move=second_move
            )

            # Generate all possible third moves using the same die value
            third_die_moves = get_moves_with_one_die(second_board, die_value, player)

            if (
                not third_die_moves
                and len(second_die_moves) == 1
                and not full_move_of_length_4_possible
            ):
                # If no third moves are possible and only one second move exists, record the sequence
                add_unique_board(
                    board=second_board,
                    moves=(first_move, second_move),
                    full_moves=full_moves,
                    unique_boards=unique_boards,
                    player=player,
                )

            for third_move in third_die_moves:
                # Apply the third move to get the new board state
                third_board = second_board.move_checker(
                    player=player, sub_move=third_move
                )

                # Generate all possible fourth moves using the same die value
                fourth_die_moves = get_moves_with_one_die(
                    third_board, die_value, player
                )

                if (
                    not fourth_die_moves
                    and len(third_die_moves) == 1
                    and not full_move_of_length_4_possible
                ):
                    # If no fourth moves are possible and only one third move exists, record the sequence
                    add_unique_board(
                        board=third_board,
                        moves=(first_move, second_move, third_move),
                        full_moves=full_moves,
                        unique_boards=unique_boards,
                        player=player,
                    )

                for fourth_move in fourth_die_moves:
                    # Apply the fourth move to get the final board state
                    final_board = third_board.move_checker(
                        player=player, sub_move=fourth_move
                    )

                    # Record the full move sequence (four sub-moves)
                    add_unique_board(
                        board=final_board,
                        moves=(first_move, second_move, third_move, fourth_move),
                        full_moves=full_moves,
                        unique_boards=unique_boards,
                        player=player,
                    )
                    full_move_of_length_4_possible = True


def add_unique_board(
    board: ImmutableBoard,
    moves: Tuple[SubMove, ...],
    full_moves: List[FullMove],
    unique_boards: Set[ImmutableBoard],
    player: Player,
):
    """
    Adds a unique full move sequence to the list of full moves.

    This function checks if the resulting board state has already been encountered by
    verifying the board against the unique_boards set. If the board state is unique,
    it adds the board to the set and appends the move sequence to full_moves.

    Args:
        board (ImmutableBoard): The resulting state of the board after applying moves.
        moves (Tuple[SubMove, ...]): The tuple of sub-moves that constitute the full move.
        full_moves (List[FullMove]): The list to store all unique full move sequences.
        unique_boards (Set[ImmutableBoard]): A set to keep track of already processed board states.
        player (Player): The player who made the moves.
    """
    if board not in unique_boards:
        unique_boards.add(board)
        # Create a new FullMove instance
        full_move = FullMove(sub_move_commands=moves, player=player)
        full_moves.append(full_move)
