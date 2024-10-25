class Episode:
    def __init__(self):
        self.experiences = []  # List of Experience objects

    def add_experience(self, experience):
        self.experiences.append(experience)


class Experience:
    def __init__(
        self,
        observation,
        action_mask,
        action,
        action_log_prob,
        state_value,
        reward,
        done,
        next_observation,
        next_state_value,
    ):
        self.observation = observation
        self.action_mask = action_mask
        self.action = action
        self.action_log_prob = action_log_prob
        self.state_value = state_value
        self.reward = reward
        self.done = done
        self.next_observation = next_observation
        self.next_state_value = next_state_value
