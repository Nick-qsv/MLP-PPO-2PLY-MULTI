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


class Episode:
    # Define a singleton padding experience to minimize memory usage
    PADDING_EXPERIENCE = Experience(
        observation=None,
        action_mask=None,
        action=None,
        action_log_prob=None,
        state_value=None,
        reward=0.0,
        done=True,
        next_observation=None,
        next_state_value=None,
    )

    def __init__(self):
        self.experiences = []  # List of Experience objects

    def add_experience(self, experience):
        self.experiences.append(experience)

    def pad_experience(self, max_len):
        current_len = len(self.experiences)
        num_padding = max_len - current_len
        if num_padding > 0:
            # Reuse the singleton padding experience to minimize memory usage
            self.experiences.extend([self.PADDING_EXPERIENCE] * num_padding)
