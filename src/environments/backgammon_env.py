import gym
from gym import spaces
import torch
import numpy as np
from typing import Dict
from backgammon.board.immutable_board import ImmutableBoard
from backgammon.moves.conditions import Player, get_opponent
from backgammon.moves.generate_all_moves import get_all_possible_moves
from .env_helper import (
    execute_full_move_on_board_copy,
    check_game_over,
    check_for_gammon,
    check_for_backgammon,
    generate_all_board_features,
)

REWARD_PASS = 0.0
REWARD_INVALID_ACTION = -1.0
REWARD_WIN_BACKGAMMON = 2.0
REWARD_WIN_GAMMON = 1.5
REWARD_WIN_NORMAL = 1.0


class BackgammonEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        worker_id=None,
        match_length=15,
        max_legal_moves=500,
        device=torch.device("cpu"),
    ):
        super(BackgammonEnv, self).__init__()

        self.match_length = match_length
        self.player_scores: Dict[int, int] = {
            Player.PLAYER1: 0,
            Player.PLAYER2: 0,
        }
        self.current_match_winner = None

        # Set the device
        self.device = device

        self.board = ImmutableBoard.initial_board()
        self.current_player = Player.PLAYER1
        self.game_over = False
        self.match_over = False

        self.max_legal_moves = max_legal_moves

        # Observation space
        board_feature_length = 198  # From get_board_features
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(board_feature_length,),
            dtype=np.float32,
        )

        # Action space
        self.action_space = spaces.Discrete(self.max_legal_moves)

        # Variables for dice roll and legal moves
        self.roll_result = None
        self.action_mask = torch.zeros(
            self.max_legal_moves, dtype=torch.float32, device=self.device
        )
        self.legal_board_features = None  # Tensor of possible next board features
        self.legal_moves = []  # List of FullMove objects

        self.worker_id = worker_id

    def reset(self):
        print(f"Worker {self.worker_id}: Entered env.reset()")

        if self.match_over:
            self.player_scores = {Player.PLAYER1: 0, Player.PLAYER2: 0}
            self.match_over = False
            self.current_match_winner = None

        # Reset the board
        self.board = ImmutableBoard.initial_board()
        self.game_over = False

        # Alternate starting player
        self.current_player = (
            Player.PLAYER1 if self.current_player == Player.PLAYER2 else Player.PLAYER2
        )
        print(f"Worker {self.worker_id}: Current player set to {self.current_player}")

        # Roll dice to determine who starts
        self.roll_dice()
        print(
            f"Worker {self.worker_id}: Initial dice roll for starting player: {self.roll_result}"
        )
        while self.roll_result[0] == self.roll_result[1]:
            self.roll_dice()

        # The player with the higher roll starts
        if self.roll_result[0] < self.roll_result[1]:
            self.current_player = Player.PLAYER2
        else:
            self.current_player = Player.PLAYER1

        # Roll dice for the first move, ensuring it's not doubles
        self.roll_dice()
        while self.roll_result[0] == self.roll_result[1]:
            self.roll_dice()

        # Update legal moves and board features based on the first non-doubles roll
        self.update_legal_moves()
        print(f"Worker {self.worker_id}: Updated legal moves")

        observation = self.get_observation()
        print(f"Worker {self.worker_id}: Exiting env.reset()")
        return observation

    def step(self, action):
        # Initialize the info dictionary with current_player
        info = {"current_player": self.current_player}

        if self.game_over:
            observation = self.get_observation()
            return observation, torch.tensor(0.0, device=self.device), True, info

        # Check if there are any legal actions
        if self.action_mask.sum() == 0:
            # No legal actions, pass the turn to the next player
            reward = torch.tensor(REWARD_PASS, device=self.device)
            done = False
            # Pass the turn
            self.pass_turn()
            self.roll_dice()
            self.update_legal_moves()

            # Get the new observation after passing the turn
            observation = self.get_observation()
            return (
                observation,
                reward,
                done,
                {**info, "info": "No legal actions, turn passed"},
            )

        # Validate action
        if not self.action_mask[action]:
            # Invalid action selected
            reward = torch.tensor(REWARD_INVALID_ACTION, device=self.device)
            done = False
            print(f"Invalid action selected: {action}. Assigned reward: {reward}")
            observation = self.get_observation()
            return observation, reward, done, {**info, "info": "Invalid action"}

        # Execute the Selected Move by applying the corresponding FullMove
        selected_move = self.legal_moves[action]
        self.board = execute_full_move_on_board_copy(self.board, selected_move)

        # Check for game over
        if check_game_over(self.board, self.current_player):
            # Winning conditions
            is_backgammon = check_for_backgammon(self.board, self.current_player)
            is_gammon = False
            if not is_backgammon:
                is_gammon = check_for_gammon(self.board, self.current_player)

            if is_backgammon:
                game_score = 3
                reward = torch.tensor(REWARD_WIN_BACKGAMMON, device=self.device)
            elif is_gammon:
                game_score = 2
                reward = torch.tensor(REWARD_WIN_GAMMON, device=self.device)
            else:
                game_score = 1
                reward = torch.tensor(REWARD_WIN_NORMAL, device=self.device)

            info.update({"winner": self.current_player, "game_score": game_score})
            self.player_scores[self.current_player] += game_score
            self.game_over = True
            done = True

            # Check if match is over
            if self.player_scores[self.current_player] >= self.match_length:
                # Match is over
                self.current_match_winner = self.current_player
                self.match_over = True
        else:
            reward = torch.tensor(0.0, device=self.device)
            done = False
            # Pass the turn to the other player
            self.pass_turn()
            self.roll_dice()
            self.update_legal_moves()

        observation = self.get_observation()
        return observation, reward, done, info

    def get_observation(self):
        # Board features
        board_features = self.board.get_board_features(self.current_player)
        return board_features.to(self.device)  # Ensure tensor is on the correct device

    def update_legal_moves_np(self):
        print(f"Worker {self.worker_id}: Entered update_legal_moves()")

        # Generate legal moves
        self.legal_moves = get_all_possible_moves(
            player=self.current_player,
            board=self.board,
            roll_result=self.roll_result,
        )
        print(f"Worker {self.worker_id}: Generated {len(self.legal_moves)} legal moves")

        # Generate legal board features for the action mask
        self.legal_board_features = (
            generate_all_board_features(
                board=self.board,
                current_player=self.current_player,
                legal_moves=self.legal_moves,
            )
            if self.legal_moves
            else torch.empty((0, 198), dtype=torch.float32, device=self.device)
        )
        print(
            f"Worker {self.worker_id}: Generated {len(self.legal_board_features)} legal board features"
        )

        num_moves = self.legal_board_features.size(0)
        if num_moves > self.max_legal_moves:
            self.legal_board_features = self.legal_board_features[
                : self.max_legal_moves, :
            ]
            self.legal_moves = self.legal_moves[: self.max_legal_moves]

        num_moves = self.legal_board_features.size(0)

        # Update action_mask
        self.action_mask = torch.zeros(
            self.max_legal_moves, dtype=torch.float32, device=self.device
        )
        self.action_mask[:num_moves] = 1.0

        # If there are fewer moves than max_legal_moves, pad the features
        if num_moves < self.max_legal_moves:
            padding_length = self.max_legal_moves - num_moves
            padding = torch.zeros(
                (padding_length, self.legal_board_features.size(1)),
                dtype=self.legal_board_features.dtype,
                device=self.device,
            )
            self.legal_board_features = torch.cat(
                [self.legal_board_features, padding], dim=0
            )

    def update_legal_moves(self):
        print(f"Worker {self.worker_id}: Entered update_legal_moves()")

        # Generate legal moves
        try:
            self.legal_moves = get_all_possible_moves(
                player=self.current_player,
                board=self.board,
                roll_result=self.roll_result,
            )
            print(
                f"Worker {self.worker_id}: Generated {len(self.legal_moves)} legal moves"
            )
        except Exception as e:
            print(f"Worker {self.worker_id}: Error in get_all_possible_moves: {e}")
            return

        # Generate legal board features for the action mask
        try:
            if self.legal_moves:
                self.legal_board_features = generate_all_board_features(
                    board=self.board,
                    current_player=self.current_player,
                    legal_moves=self.legal_moves,
                )
                print(f"Worker {self.worker_id}: Generated legal board features")
            else:
                self.legal_board_features = torch.empty(
                    (0, 198), dtype=torch.float32, device=self.device
                )
                print(
                    f"Worker {self.worker_id}: No legal moves, generated empty board features"
                )
            print(
                f"Worker {self.worker_id}: Legal board features shape: {self.legal_board_features.shape}"
            )
        except Exception as e:
            print(f"Worker {self.worker_id}: Error in generate_all_board_features: {e}")
            return

        # Truncate legal moves and board features if they exceed max_legal_moves
        num_moves = self.legal_board_features.size(0)
        if num_moves > self.max_legal_moves:
            print(f"Worker {self.worker_id}: Truncating legal moves and board features")
            self.legal_board_features = self.legal_board_features[
                : self.max_legal_moves, :
            ]
            self.legal_moves = self.legal_moves[: self.max_legal_moves]
        print(
            f"Worker {self.worker_id}: Final number of legal moves: {len(self.legal_moves)}"
        )

        # Update action_mask
        try:
            print(f"Worker {self.worker_id}: self.device = {self.device}")
            self.action_mask = torch.zeros(
                self.max_legal_moves, dtype=torch.float32, device=self.device
            )
            self.action_mask[:num_moves] = 1.0
            print(f"Worker {self.worker_id}: Action mask updated")
        except Exception as e:
            print(f"Worker {self.worker_id}: Error updating action_mask: {e}")
            return

        # If fewer moves than max_legal_moves, pad the features
        try:
            if num_moves < self.max_legal_moves:
                padding_length = self.max_legal_moves - num_moves
                print(
                    f"Worker {self.worker_id}: num_moves = {num_moves}, padding_length = {padding_length}"
                )

                # Check devices
                print(f"Worker {self.worker_id}: self.device = {self.device}")
                print(
                    f"Worker {self.worker_id}: self.legal_board_features.device = {self.legal_board_features.device}"
                )

                # Ensure legal_board_features is on the correct device
                if self.legal_board_features.device != torch.device("cpu"):
                    print(
                        f"Worker {self.worker_id}: Moving legal_board_features to CPU"
                    )
                    self.legal_board_features = self.legal_board_features.to("cpu")

                padding = torch.zeros(
                    (padding_length, self.legal_board_features.size(1)),
                    dtype=self.legal_board_features.dtype,
                    device="cpu",
                )
                print(f"Worker {self.worker_id}: padding.device = {padding.device}")

                self.legal_board_features = torch.cat(
                    [self.legal_board_features, padding], dim=0
                )
                print(
                    f"Worker {self.worker_id}: Padded legal board features to max_legal_moves"
                )
        except Exception as e:
            print(f"Worker {self.worker_id}: Error during padding: {e}")
            return

        print(f"Worker {self.worker_id}: Exiting update_legal_moves()")

    def roll_dice(self):
        self.roll_result = [np.random.randint(1, 7), np.random.randint(1, 7)]

    def pass_turn(self):
        self.current_player = get_opponent(self.current_player)
