import logging
from dataclasses import dataclass
from typing import Tuple
from src.backgammon.types import Position, Player, SubMove

logging.basicConfig(
    level=logging.WARNING,  # Change to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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

    def __hash__(self):
        """
        Efficient hash function for ImmutableBoard by combining all board attributes.
        """
        # Flatten all positions and combine them into a single tuple for hashing
        return hash(
            (
                self.positions_0,
                self.positions_1,
                self.bar,
                self.borne_off,
            )
        )

    def move_checker(self, player: Player, sub_move: SubMove) -> "ImmutableBoard":
        positions_0, positions_1 = list(self.positions_0), list(self.positions_1)
        bar, borne_off = list(self.bar), list(self.borne_off)

        player_idx = player.value
        opponent_idx = 1 - player_idx
        start, end, hits_blot = sub_move.start, sub_move.end, sub_move.hits_blot

        # Remove checker from start
        if start == Position.BAR:
            if bar[player_idx] <= 0:
                logger.warning(
                    "No checker to remove from bar for Player %d.", player_idx + 1
                )
                return self
            bar[player_idx] -= 1
            logger.debug("Removed 1 checker from Player %d's bar.", player_idx + 1)
        else:
            player_positions = positions_0 if player_idx == 0 else positions_1
            if player_positions[start.value] <= 0:
                logger.warning(
                    "No checker to remove at point %s for Player %d.",
                    start.name,
                    player_idx + 1,
                )
                return self
            player_positions[start.value] -= 1
            logger.debug(
                "Removed 1 checker from Player %d's point %s.",
                player_idx + 1,
                start.name,
            )

        # Handle hitting a blot
        if hits_blot:
            opponent_positions = positions_0 if opponent_idx == 0 else positions_1
            if opponent_positions[end.value] == 1:
                opponent_positions[end.value] -= 1
                bar[opponent_idx] += 1
                logger.debug(
                    "Player %d hit a blot at %s. Player %d's checker sent to the bar.",
                    player_idx + 1,
                    end.name,
                    opponent_idx + 1,
                )
            else:
                logger.warning(
                    "No blot to hit at point %s for Player %d.",
                    end.name,
                    player_idx + 1,
                )
                return self

        # Add checker to end
        if end == Position.BEAR_OFF:
            borne_off[player_idx] += 1
            logger.debug(
                "Player %d borne off a checker. Total borne off: %d",
                player_idx + 1,
                borne_off[player_idx],
            )
        else:
            player_positions[end.value] += 1
            logger.debug(
                "Added 1 checker to Player %d's point %s.", player_idx + 1, end.name
            )

        return ImmutableBoard(
            positions_0=tuple(positions_0),
            positions_1=tuple(positions_1),
            bar=tuple(bar),
            borne_off=tuple(borne_off),
        )


# 4. Enhancing Episode Grouping for Clarity and Efficiency

# To improve clarity and ensure efficient processing, you might consider explicitly grouping experiences by episodes. Here’s how you can modify your implementation:
# a. Modify Memory Structure to Store Episodes Separately
# Instead of using a flat list, use a list of episodes, where each episode is a list of experiences.
# c. Adjust the Agent's Update Method to Handle Grouped Episodes
# d. Benefits of Explicit Episode Grouping
# Clarity: Clearly delineates where one episode ends and another begins, simplifying debugging and analysis.
# Flexible Processing: Facilitates more nuanced processing of episodes, such as applying episode-specific transformations or analysis.
# Enhanced Credit Assignment: Makes it easier to compute returns and advantages on a per-episode basis, ensuring accurate credit assignment even in sparse reward settings.
