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

    def to_tensor(self):
        for attr in vars(self):
            val = getattr(self, attr)
            if isinstance(val, np.ndarray):
                setattr(self, attr, torch.from_numpy(val))
            elif isinstance(val, float):
                setattr(self, attr, torch.tensor(val, dtype=torch.float32))
            elif isinstance(val, int):
                setattr(self, attr, torch.tensor(val, dtype=torch.int64))
            elif isinstance(val, bool):
                setattr(self, attr, torch.tensor(val, dtype=torch.bool))
            elif val is not None and hasattr(val, "to_tensor"):
                val.to_tensor()


class Episode:
    def __init__(self):
        self.experiences = []  # List of Experience objects

    def add_experience(self, experience):
        self.experiences.append(experience)

    def to_numpy(self):
        for experience in self.experiences:
            experience.to_numpy()

    def to_tensor(self):
        for experience in self.experiences:
            experience.to_tensor()
