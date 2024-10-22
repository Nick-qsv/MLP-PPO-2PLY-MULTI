from src.backgammon.types import Player, NUMBER_OF_POINTS, Position, BoardState
from src.backgammon.board import ImmutableBoard


def compute_board_state(board: ImmutableBoard, player: Player) -> "BoardState":
    """
    Determines the current state of the board for the specified player.

    Args:
        board (ImmutableBoard): The current state of the board.
        player (Player): The player for whom to compute the board state.

    Returns:
        BoardState: The current state of the board (NORMAL, ON_BAR, BEAR_OFF, GAME_OVER).
    """
    if check_for_win(board, player):
        return BoardState.GAME_OVER  # Player has won the game
    if check_for_bar(board, player):
        return BoardState.ON_BAR  # Player has checkers on the bar
    if all_checkers_home(board, player):
        return BoardState.BEAR_OFF  # Player can start bearing off
    return BoardState.NORMAL  # Player is in a normal state


def valid_move(destination_idx: int, player: Player, board: ImmutableBoard) -> bool:
    """
    Checks if moving to the destination index is a valid move for the player on the given ImmutableBoard.

    Args:
        destination_idx (int): The index of the destination point.
        player (Player): The player making the move.
        board (ImmutableBoard): The current state of the board.

    Returns:
        bool: True if the move is valid, False otherwise.
    """
    player_idx = player.value  # 0 for PLAYER1, 1 for PLAYER2
    opponent_idx = 1 - player_idx  # 1 for PLAYER1, 0 for PLAYER2

    # Check if destination is within normal points
    if 0 <= destination_idx < NUMBER_OF_POINTS:
        # Get opponent's checkers at the destination point
        if opponent_idx == Player.PLAYER1.value:
            opponent_checkers = board.positions_0[destination_idx]
        else:
            opponent_checkers = board.positions_1[destination_idx]

        # If opponent has 2 or more checkers, the point is blocked
        if opponent_checkers >= 2:
            return False
        else:
            return True

    # Check if the move is bearing off
    elif destination_idx == Position.BEAR_OFF.value:
        # Bearing off rules can be more complex; here we assume it's allowed
        # You might want to add additional checks based on your game rules
        return True

    else:
        # Invalid destination index
        return False


def check_if_blot(index: int, player: Player, board: ImmutableBoard) -> bool:
    """
    Checks if there is a blot (exactly one opponent checker) at the specified index.

    Args:
        index (int): The point index to check for a blot.
        player (Player): The current player.
        board (ImmutableBoard): The current state of the board.

    Returns:
        bool: True if there is a blot at the index, False otherwise.
    """
    opponent_idx = 1 - player.value  # Opponent's index

    if 0 <= index < NUMBER_OF_POINTS:
        # Retrieve the number of opponent's checkers at the specified index
        if opponent_idx == Player.PLAYER1.value:
            opponent_checkers = board.positions_0[index]
        else:
            opponent_checkers = board.positions_1[index]

        # A blot is exactly one opponent checker
        if opponent_checkers == 1:
            return True  # There is a blot
    return False


def is_valid_entry_at_index(index: int, player: Player, board: ImmutableBoard) -> bool:
    """
    Determines if the player can enter a checker at the specified index.

    Args:
        index (int): The point index where the checker is to be entered.
        player (Player): The player attempting to enter the checker.
        board (ImmutableBoard): The current state of the board.

    Returns:
        bool: True if the entry is valid, False otherwise.
    """
    opponent_idx = 1 - player.value  # Opponent's index

    if 0 <= index < NUMBER_OF_POINTS:
        # Retrieve the number of opponent's checkers at the specified index
        if opponent_idx == Player.PLAYER1.value:
            opponent_checkers = board.positions_0[index]
        else:
            opponent_checkers = board.positions_1[index]

        # If the opponent has 2 or more checkers, the entry point is blocked
        if opponent_checkers >= 2:
            return False  # Entry point is blocked
        else:
            return True  # Entry point is open or has a blot
    else:
        return False  # Invalid index for entry


def check_for_bar(board: ImmutableBoard, player: Player) -> bool:
    """
    Checks if the player has any checkers on the bar.

    Args:
        board (ImmutableBoard): The current state of the board.
        player (Player): The player to check for checkers on the bar.

    Returns:
        bool: True if the player has checkers on the bar, False otherwise.
    """
    # Access the bar tuple using the player's index and check if greater than 0
    return board.bar[player.value] > 0


def check_for_win(board: ImmutableBoard, player: Player) -> bool:
    """
    Checks if the player has borne off all checkers, indicating a win.

    Args:
        board (ImmutableBoard): The current state of the board.
        player (Player): The player to check for a win.

    Returns:
        bool: True if the player has won, False otherwise.
    """
    # Access the borne_off tuple using the player's index and check if it equals 15
    return board.borne_off[player.value] == 15


def all_checkers_home(board: ImmutableBoard, player: Player) -> bool:
    """
    Determines if all of the player's checkers are in the home board or have been borne off.

    Args:
        board (ImmutableBoard): The current state of the board.
        player (Player): The player to check.

    Returns:
        bool: True if all checkers are home or borne off, False otherwise.
    """
    player_idx = player.value  # Player's index

    # Check for any checkers on the bar
    if board.bar[player_idx] > 0:
        return False  # Checkers are on the bar

    # Define home range based on player
    if player == Player.PLAYER2:
        home_range = range(0, 6)  # Points 0-5 for PLAYER2
    else:
        home_range = range(18, 24)  # Points 18-23 for PLAYER1

    total_checkers = 0
    # Retrieve the player's positions
    if player == Player.PLAYER1:
        player_positions = board.positions_0
    else:
        player_positions = board.positions_1

    for idx in range(NUMBER_OF_POINTS):  # Iterate over all points
        num_checkers = player_positions[idx]
        if num_checkers > 0:
            if idx in home_range:
                total_checkers += num_checkers
            else:
                return False  # Checker is outside the home board

    # Include borne-off checkers
    borne_off_checkers = board.borne_off[player_idx]

    # Total checkers should be 15 (home + borne off)
    return total_checkers + borne_off_checkers == 15
