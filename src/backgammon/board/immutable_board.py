import logging
import numpy as np
import torch
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
    positions_0: Tuple[int, ...]  # Length 24 (Player 1's checkers)
    positions_1: Tuple[int, ...]  # Length 24 (Player 2's checkers)
    # Store the number of checkers on the bar and borne off for each player
    bar: Tuple[int, int]  # (Player1_bar, Player2_bar)
    borne_off: Tuple[int, int]  # (Player1_borne_off, Player2_borne_off)
    device: torch.device = torch.device("cpu")  # Device to place tensors on

    @staticmethod
    def initial_board(device: torch.device = torch.device("cpu")) -> "ImmutableBoard":
        # Initialize positions with zeros
        positions_0 = [0] * 24
        positions_1 = [0] * 24

        # # Starting Board Player 1
        # positions_0[Position.P_0] = 2
        # positions_0[Position.P_11] = 5
        # positions_0[Position.P_16] = 3
        # positions_0[Position.P_18] = 5

        # # Starting Board Player 2
        # positions_1[Position.P_23] = 2
        # positions_1[Position.P_12] = 5
        # positions_1[Position.P_7] = 3
        # positions_1[Position.P_5] = 5

        # Starting Board P1 Prime/Closeout Tutor
        positions_0[Position.P_0] = 2
        positions_0[Position.P_11] = 4
        positions_0[Position.P_13] = 1
        positions_0[Position.P_14] = 1
        positions_0[Position.P_15] = 1
        positions_0[Position.P_17] = 2
        positions_0[Position.P_18] = 2
        positions_0[Position.P_19] = 2

        # Starting Board P2 Prime/Closeout Tutor
        positions_1[Position.P_23] = 2
        positions_1[Position.P_12] = 4
        positions_1[Position.P_10] = 1
        positions_1[Position.P_9] = 1
        positions_1[Position.P_8] = 1
        positions_1[Position.P_6] = 2
        positions_1[Position.P_5] = 2
        positions_1[Position.P_4] = 2

        return ImmutableBoard(
            positions_0=tuple(positions_0),
            positions_1=tuple(positions_1),
            bar=(0, 0),
            borne_off=(0, 0),
            device=device,
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

    def get_board_features(self, current_player: Player) -> torch.Tensor:
        """
        Computes the 198-dimensional feature vector for the board state.
        Returns:
            torch.Tensor: A tensor of shape (198,) representing the board features.
        """
        # Preallocate the feature tensor
        features = torch.zeros(198, dtype=torch.float32, device=self.device)

        # Combine positions of both players into a single tensor of shape (2, 24)
        positions = torch.tensor(
            [self.positions_0, self.positions_1], dtype=torch.int8, device=self.device
        )  # Shape: (2, 24)

        # Compute features:
        # - Presence of at least 1, 2, 3 checkers
        # - Number of checkers beyond 3, clamped and scaled
        presence_1 = (positions >= 1).float()  # Shape: (2, 24)
        presence_2 = (positions >= 2).float()  # Shape: (2, 24)
        presence_3 = (positions >= 3).float()  # Shape: (2, 24)
        excess = torch.clamp(positions - 3, min=0).float() / 2.0  # Shape: (2, 24)

        # Stack all features along a new dimension and reshape
        feature_stack = torch.stack(
            [presence_1, presence_2, presence_3, excess], dim=2
        )  # Shape: (2, 24, 4)
        features[:192] = feature_stack.view(-1)  # Flatten to (192,)

        # Assign bar and borne off features for both players
        # Player 1
        features[192] = self.bar[Player.PLAYER1] / 2.0
        features[193] = self.borne_off[Player.PLAYER1] / 15.0
        # Player 2
        features[194] = self.bar[Player.PLAYER2] / 2.0
        features[195] = self.borne_off[Player.PLAYER2] / 15.0

        # Assign current player indicator features
        is_player1 = current_player == Player.PLAYER1
        is_player2 = current_player == Player.PLAYER2
        features[196] = 1.0 if is_player1 else 0.0
        features[197] = 1.0 if is_player2 else 0.0

        return features

    def get_board_features_old(self, current_player: Player) -> torch.Tensor:
        """
        Computes the 198-dimensional feature vector for the board state.
        Returns:
            torch.Tensor: A tensor of shape (198,) representing the board features.
        """
        features = torch.zeros(198, dtype=torch.float32, device=self.device)
        feature_index = 0

        # Process both players' positions
        for player_idx, positions in enumerate([self.positions_0, self.positions_1]):
            # Convert positions to torch tensor for vectorized operations
            positions_tensor = torch.tensor(
                positions, dtype=torch.int8, device=self.device
            )

            # Initialize a (24, 4) feature slice for points
            features_slice = torch.zeros(
                (24, 4), dtype=torch.float32, device=self.device
            )

            # Vectorized computation for point features
            checkers = positions_tensor
            features_slice[:, 0] = (checkers >= 1).float()
            features_slice[:, 1] = (checkers >= 2).float()
            features_slice[:, 2] = (checkers >= 3).float()
            features_slice[:, 3] = torch.clamp(checkers - 3, min=0).float() / 2.0

            # Flatten and assign to the main feature vector
            features[feature_index : feature_index + 96] = features_slice.view(-1)
            feature_index += 96

            # Bar checkers feature
            bar_checkers = self.bar[player_idx]
            features[feature_index] = bar_checkers / 2.0
            feature_index += 1

            # Borne off checkers feature
            borne_off_checkers = self.borne_off[player_idx]
            features[feature_index] = borne_off_checkers / 15.0
            feature_index += 1

        # Current player indicator features
        features[feature_index] = 1.0 if current_player == Player.PLAYER1 else 0.0
        features[feature_index + 1] = 1.0 if current_player == Player.PLAYER2 else 0.0
        feature_index += 2

        assert (
            feature_index == 198
        ), f"Feature vector length is {feature_index}, expected 198"

        return features

    def move_checker(self, player: Player, sub_move: SubMove) -> "ImmutableBoard":
        positions_0, positions_1 = list(self.positions_0), list(self.positions_1)
        bar, borne_off = list(self.bar), list(self.borne_off)

        player_idx = player.value
        opponent_idx = 1 - player_idx
        start, end, hits_blot = sub_move.start, sub_move.end, sub_move.hits_blot

        # Assign player_positions for the current player
        player_positions = positions_0 if player_idx == 0 else positions_1
        opponent_positions = positions_0 if opponent_idx == 0 else positions_1

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
            device=self.device,
        )
