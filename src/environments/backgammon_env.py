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
    is_closed_out,
    made_at_least_five_prime,
)
import time

REWARD_PASS = 0.0
REWARD_INVALID_ACTION = -1.0
REWARD_WIN_BACKGAMMON = 2.2
REWARD_WIN_GAMMON = 2.0
REWARD_WIN_NORMAL = 1.0
REWARD_CLOSE_OUT = 0.36
REWARD_MAKE_PRIME = 0.24


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
        self.current_board_features = self.board.get_board_features(self.current_player)
        self.max_legal_moves = max_legal_moves

        # Observation space
        board_feature_length = 198  # From get_board_features
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(board_feature_length,),
            dtype=np.float32,
        )
        self.board_feature_length = board_feature_length  # Save for later use

        # Action space
        self.action_space = spaces.Discrete(self.max_legal_moves)

        # Variables for dice roll and legal moves
        self.roll_result = None
        # Preallocate tensors for action mask and legal board features
        self.action_mask = torch.zeros(
            self.max_legal_moves, dtype=torch.float32, device=self.device
        )
        self.legal_board_features = torch.zeros(
            (self.max_legal_moves, board_feature_length),
            dtype=torch.float32,
            device=self.device,
        )
        self.legal_moves = []  # List of FullMove objects
        self.num_moves = 0
        self.previous_num_moves = 0

        self.worker_id = worker_id
        self.profiling_data = {}

        self.win_type = None

        # Initialize reward tracking flags for each player
        self.close_out_reward_given = {
            Player.PLAYER1: False,
            Player.PLAYER2: False,
        }
        self.prime_reward_given = {
            Player.PLAYER1: False,
            Player.PLAYER2: False,
        }

    def reset(self):

        if self.match_over:
            self.player_scores = {Player.PLAYER1: 0, Player.PLAYER2: 0}
            self.match_over = False
            self.current_match_winner = None

        # Reset the board using setter method
        self.set_board(ImmutableBoard.initial_board())
        self.game_over = False
        self.win_type = None

        # Alternate starting player using setter method
        self.set_current_player(
            Player.PLAYER1 if self.current_player == Player.PLAYER2 else Player.PLAYER2
        )

        # Roll dice to determine who starts
        self.roll_dice()
        while self.roll_result[0] == self.roll_result[1]:
            self.roll_dice()

        # The player with the higher roll starts
        if self.roll_result[0] < self.roll_result[1]:
            self.set_current_player(Player.PLAYER2)
        else:
            self.set_current_player(Player.PLAYER1)

        # Roll dice for the first move, ensuring it's not doubles
        self.roll_dice()
        while self.roll_result[0] == self.roll_result[1]:
            self.roll_dice()

        # Update legal moves and board features based on the first non-doubles roll
        self.update_legal_moves()

        # Reset reward tracking flags for the new game
        self.close_out_reward_given = {
            Player.PLAYER1: False,
            Player.PLAYER2: False,
        }
        self.prime_reward_given = {
            Player.PLAYER1: False,
            Player.PLAYER2: False,
        }

        observation = self.get_observation()
        return observation

    def step(self, action):
        # Initialize the info dictionary with current_player
        info = {"current_player": self.current_player}

        if self.game_over:
            observation = self.get_observation()
            return observation, torch.tensor(0.0, device=self.device), True, info

        # Check if there are any legal actions
        if self.action_mask.sum() == 0:
            reward = torch.tensor(REWARD_PASS, device=self.device)
            done = False
            self.pass_turn()
            self.roll_dice()
            self.profile_call(self.update_legal_moves)
            observation = self.get_observation()
            return (
                observation,
                reward,
                done,
                {**info, "info": "No legal actions, turn passed"},
            )

        if not self.action_mask[action].item():
            reward = torch.tensor(REWARD_INVALID_ACTION, device=self.device)
            print(f"Worker {self.worker_id}: Invalid action {action}")
            done = False
            observation = self.get_observation()
            return observation, reward, done, {**info, "info": "Invalid action"}

        selected_move = self.legal_moves[action]
        # Update the board using setter method
        self.set_board(execute_full_move_on_board_copy(self.board, selected_move))

        # Initialize reward to zero
        reward = torch.tensor(0.0, device=self.device)

        if check_game_over(self.board, self.current_player):
            # Process game over rewards
            is_backgammon = check_for_backgammon(self.board, self.current_player)
            is_gammon = False
            win_type = "regular"  # Default win type

            if is_backgammon:
                game_score = 3
                reward = torch.tensor(REWARD_WIN_BACKGAMMON, device=self.device)
                win_type = "backgammon"
            else:
                is_gammon = check_for_gammon(self.board, self.current_player)
                if is_gammon:
                    game_score = 2
                    reward = torch.tensor(REWARD_WIN_GAMMON, device=self.device)
                    win_type = "gammon"
                else:
                    game_score = 1
                    reward = torch.tensor(REWARD_WIN_NORMAL, device=self.device)

            info.update(
                {
                    "winner": self.current_player,
                    "game_score": game_score,
                    "win_type": win_type,  # Include win type in info
                }
            )
            self.win_type = win_type  # Set the win_type attribute

            self.player_scores[self.current_player] += game_score
            self.game_over = True
            done = True

            if self.player_scores[self.current_player] >= self.match_length:
                self.current_match_winner = self.current_player
                self.match_over = True
        else:
            # Game not over, check for close-out and prime
            # Check for close out
            if (
                is_closed_out(self.board, self.current_player)
                and not self.close_out_reward_given[self.current_player]
            ):
                reward += torch.tensor(REWARD_CLOSE_OUT, device=self.device)
                self.close_out_reward_given[self.current_player] = True
                info["close_out_reward"] = True

            # Check for making at least a 5-prime
            if (
                made_at_least_five_prime(self.board, self.current_player)
                and not self.prime_reward_given[self.current_player]
            ):
                reward += torch.tensor(REWARD_MAKE_PRIME, device=self.device)
                self.prime_reward_given[self.current_player] = True
                info["prime_reward"] = True

            done = False
            self.pass_turn()
            self.roll_dice()
            self.update_legal_moves()

        observation = self.get_observation()
        return observation, reward, done, info

    def update_legal_moves(self):
        # Generate legal moves
        try:
            self.legal_moves = get_all_possible_moves(
                player=self.current_player,
                board=self.board,
                roll_result=self.roll_result,
            )
        except Exception as e:
            print(f"Worker {self.worker_id}: Error in get_all_possible_moves: {e}")
            return

        # Generate legal board features
        try:
            if self.legal_moves:
                legal_board_features = generate_all_board_features(
                    board=self.board,
                    current_player=self.current_player,
                    legal_moves=self.legal_moves,
                )
            else:
                legal_board_features = torch.empty(
                    (0, self.board_feature_length), dtype=torch.float32
                )
        except Exception as e:
            print(f"Worker {self.worker_id}: Error in generate_all_board_features: {e}")
            return

        # Refactored: Truncate legal moves and board features
        legal_board_features = self.truncate_legal_moves_and_features(
            legal_board_features
        )

        # Refactored: Update action_mask
        self.update_action_mask()

        # Refactored: Update legal_board_features
        self.update_legal_board_features(legal_board_features)

    def truncate_legal_moves_and_features(self, legal_board_features):
        """
        Truncate legal moves and board features if they exceed max_legal_moves.
        """
        num_moves = legal_board_features.size(0)
        if num_moves > self.max_legal_moves:
            legal_board_features = legal_board_features[: self.max_legal_moves]
            self.legal_moves = self.legal_moves[: self.max_legal_moves]
            num_moves = self.max_legal_moves
        self.num_moves = num_moves
        return legal_board_features

    def update_action_mask(self):
        """
        Update the action mask tensor based on the number of legal moves without resetting the entire tensor.
        """
        try:
            # Ensure self.num_moves does not exceed self.max_legal_moves
            current_num_moves = min(self.num_moves, self.max_legal_moves)

            # Determine if num_moves has increased or decreased
            if current_num_moves > self.previous_num_moves:
                # Set the new valid actions to 1.0
                self.action_mask[self.previous_num_moves : current_num_moves].fill_(1.0)
            elif current_num_moves < self.previous_num_moves:
                # Reset the now-invalid actions back to 0.0
                self.action_mask[current_num_moves : self.previous_num_moves].zero_()
            # If current_num_moves == previous_num_moves, do nothing

            # Update the previous_num_moves for the next call
            self.previous_num_moves = current_num_moves

        except Exception as e:
            print(f"Worker {self.worker_id}: Error updating action_mask: {e}")
            return

    def update_legal_board_features(self, legal_board_features):
        """
        Update the legal_board_features tensor with the provided features.
        """
        try:
            if self.num_moves > 0:
                self.legal_board_features[: self.num_moves].copy_(legal_board_features)
            # Removed the zeroing operation to improve performance
        except Exception as e:
            print(f"Worker {self.worker_id}: Error updating legal_board_features: {e}")
            return

    def roll_dice(self):
        self.roll_result = [np.random.randint(1, 7), np.random.randint(1, 7)]

    def get_observation(self):
        # Return the cached board features
        return self.current_board_features

    def set_board(self, new_board):
        self.board = new_board
        # Update cached board features
        self.current_board_features = self.board.get_board_features(self.current_player)

    def set_current_player(self, new_player):
        self.current_player = new_player
        # Update cached board features
        self.current_board_features = self.board.get_board_features(self.current_player)

    def pass_turn(self):
        # Use setter method to update current player
        self.set_current_player(get_opponent(self.current_player))

    def profile_call(self, func, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        func_name = func.__name__
        if func_name not in self.profiling_data:
            self.profiling_data[func_name] = {"total_time": 0.0, "call_count": 0}
        self.profiling_data[func_name]["total_time"] += elapsed
        self.profiling_data[func_name]["call_count"] += 1
        return result
