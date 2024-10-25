from collections import deque


class RingReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)  # Stores Episode objects

    def add_episode(self, episode):
        self.buffer.append(episode)

    def sample_episodes(self, num_episodes):
        # Returns a list of sampled episodes
        pass  # Sampling logic goes here
