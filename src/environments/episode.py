import torch
import numpy as np


class Experience:
    def __init__(
        self,
        observation,
        state_value,
        reward,
        done,
        next_observation,
        next_state_value,
    ):
        self.observation = observation
        self.state_value = state_value
        self.reward = reward
        self.done = done
        self.next_observation = next_observation
        self.next_state_value = next_state_value

    def to_numpy(self):
        for attr in vars(self):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.cpu().numpy())
            elif val is not None and hasattr(val, "to_numpy"):
                val.to_numpy()

    def to_tensor(self, device=None):
        for attr in vars(self):
            val = getattr(self, attr)
            if isinstance(val, np.ndarray):
                setattr(self, attr, torch.from_numpy(val).to(device))
            elif isinstance(val, float):
                setattr(
                    self, attr, torch.tensor(val, dtype=torch.float32, device=device)
                )
            elif isinstance(val, int):
                setattr(self, attr, torch.tensor(val, dtype=torch.int64, device=device))
            elif isinstance(val, bool):
                setattr(self, attr, torch.tensor(val, dtype=torch.bool, device=device))
            elif isinstance(val, torch.Tensor) and device is not None:
                setattr(self, attr, val.to(device))
            elif val is not None and hasattr(val, "to_tensor"):
                val.to_tensor(device=device)


class Episode:
    def __init__(self):
        self.experiences = []  # List of Experience objects
        self.win_type = None  # Initialize win_type
        self.close_out_counts = {}  # Initialize close out counts per player
        self.prime_reward_counts = {}  # Initialize prime reward counts per player

    def add_experience(self, experience, info):
        self.experiences.append(experience)

        # Update win_type if present
        if info.get("win_type"):
            self.win_type = info["win_type"]

        # Get current player
        current_player = info.get("current_player", None)
        if current_player is not None:
            # Initialize counts if not present
            if current_player not in self.close_out_counts:
                self.close_out_counts[current_player] = 0
            if current_player not in self.prime_reward_counts:
                self.prime_reward_counts[current_player] = 0

            # Increment counts based on info flags
            if info.get("close_out_reward", False):
                self.close_out_counts[current_player] += 1
            if info.get("prime_reward", False):
                self.prime_reward_counts[current_player] += 1

    def to_numpy(self):
        for experience in self.experiences:
            experience.to_numpy()

    def to_tensor(self, device=None):
        for experience in self.experiences:
            experience.to_tensor(device=device)
