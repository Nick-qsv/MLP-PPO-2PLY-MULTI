from typing import Tuple, List, Set
from src.backgammon.board.immutable_board import ImmutableBoard
from src.backgammon.types import SubMove, FullMove, Player


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
