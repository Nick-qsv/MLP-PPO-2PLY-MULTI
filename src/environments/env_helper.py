from backgammon.types.moves import Player, FullMove
from backgammon.board.immutable_board import ImmutableBoard


def get_opponent(player: Player) -> Player:
    return Player.PLAYER2 if player == Player.PLAYER1 else Player.PLAYER1


def execute_full_move_on_board_copy(
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
