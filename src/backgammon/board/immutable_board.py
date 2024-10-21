from dataclasses import dataclass
from typing import Tuple
from src.backgammon.types import Position


@dataclass(frozen=True)
class ImmutableBoard:
    # For each point on the board, store the number of checkers for each player
    positions_0: Tuple[int, ...]  # Length 24
    positions_1: Tuple[int, ...]  # Length 24
    # Store the number of checkers on the bar and borne off for each player
    bar: Tuple[int, int]  # (Player1_bar, Player2_bar)
    borne_off: Tuple[int, int]  # (Player1_borne_off, Player2_borne_off)

    @staticmethod
    def initial_board() -> "ImmutableBoard":
        # Initialize positions with zeros
        positions_0 = [0] * 24
        positions_1 = [0] * 24

        # Set initial positions for Player 1
        positions_0[Position.P_0] = 2
        positions_0[Position.P_11] = 5
        positions_0[Position.P_16] = 3
        positions_0[Position.P_18] = 5

        # Set initial positions for Player 2
        positions_1[Position.P_23] = 2
        positions_1[Position.P_12] = 5
        positions_1[Position.P_7] = 3
        positions_1[Position.P_5] = 5

        return ImmutableBoard(
            positions_0=tuple(positions_0),
            positions_1=tuple(positions_1),
            bar=(0, 0),
            borne_off=(0, 0),
        )
