import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import multiprocessing

# Hyperparameters
# GAMMA = 0.99
# LAMBDA = 0.95
# LEARNING_RATE = 3e-4
# ENTROPY_COEF_START = 0.01
# ENTROPY_COEF_END = 0.001
# ENTROPY_ANNEAL_EPISODES = 100000  # Number of episodes over which to anneal entropy coefficient
# EPSILON = 0.2  # PPO clipping parameter
# K_EPOCHS = 4  # Number of epochs to update policy
# BATCH_SIZE = 64  # Batch size for mini-batch updates


class Trainer:
    def __init__(self, parameter_manager, trainer_queue, device=None):
        self.parameter_manager = parameter_manager
        self.trainer_queue = trainer_queue
        self.device = device if device is not None else torch.device("cpu")

        # Initialize policy network
        self.policy_network = BackgammonPolicyNetwork().to(self.device)
        # Load initial parameters from parameter manager
        state_dict = self.parameter_manager.get_parameters()
        self.policy_network.load_state_dict(state_dict)
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=LEARNING_RATE
        )

        # Hyperparameters
        self.gamma = GAMMA
        self.lamda = LAMBDA
        self.entropy_coef_start = ENTROPY_COEF_START
        self.entropy_coef_end = ENTROPY_COEF_END
        self.entropy_anneal_episodes = ENTROPY_ANNEAL_EPISODES
        self.total_episodes = 0
        self.entropy_coef = self.entropy_coef_start

        # PPO parameters
        self.epsilon = EPSILON
        self.K_epochs = K_EPOCHS
        self.batch_size = BATCH_SIZE

    def train(self):
        while True:
            # Wait for episodes to arrive
            episodes = self.trainer_queue.get()
            # Update the total episodes
            self.total_episodes += len(episodes)
            # Process episodes
            self.update(episodes)
            # Update entropy coefficient
            self.update_entropy_coef()
            # After update, update parameters in parameter manager
            self.parameter_manager.update_parameters(self.policy_network.state_dict())
            self.parameter_manager.increment_version()

    def update(self, episodes):
        # Collect experiences from episodes
        observations = []
        actions = []
        action_log_probs = []
        rewards = []
        dones = []
        state_values = []
        next_state_values = []
        action_masks = []
        advantages = []

        for episode in episodes:
            # For each episode, extract experiences
            episode_observations = []
            episode_actions = []
            episode_action_log_probs = []
            episode_rewards = []
            episode_dones = []
            episode_state_values = []
            episode_next_state_values = []
            episode_action_masks = []

            for experience in episode.experiences:
                episode_observations.append(experience.observation)
                episode_actions.append(
                    experience.action if experience.action is not None else -1
                )
                episode_action_log_probs.append(
                    experience.action_log_prob
                    if experience.action_log_prob is not None
                    else torch.tensor(0.0)
                )
                episode_rewards.append(experience.reward)
                episode_dones.append(experience.done)
                episode_state_values.append(
                    experience.state_value
                    if experience.state_value is not None
                    else torch.tensor(0.0)
                )
                episode_next_state_values.append(
                    experience.next_state_value
                    if experience.next_state_value is not None
                    else torch.tensor(0.0)
                )
                episode_action_masks.append(experience.action_mask)

            # Compute GAE for this episode
            gae = 0
            episode_advantages = []
            for i in reversed(range(len(episode_rewards))):
                delta = (
                    episode_rewards[i]
                    + self.gamma * episode_next_state_values[i] * (1 - episode_dones[i])
                    - episode_state_values[i]
                )
                gae = delta + self.gamma * self.lamda * (1 - episode_dones[i]) * gae
                episode_advantages.insert(0, gae)
                if episode_dones[i]:
                    gae = 0
            advantages.extend(episode_advantages)

            # Append episode data to overall lists
            observations.extend(episode_observations)
            actions.extend(episode_actions)
            action_log_probs.extend(episode_action_log_probs)
            rewards.extend(episode_rewards)
            dones.extend(episode_dones)
            state_values.extend(episode_state_values)
            next_state_values.extend(episode_next_state_values)
            action_masks.extend(episode_action_masks)

        # Convert lists to tensors
        observations = torch.stack(observations).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        action_log_probs = torch.stack(action_log_probs).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        state_values = torch.stack(state_values).to(self.device)
        next_state_values = torch.stack(next_state_values).to(self.device)
        action_masks = torch.stack(action_masks).to(self.device)
        advantages = torch.tensor(advantages).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_action_log_probs = action_log_probs.detach()
        old_actions = actions.detach()

        # PPO policy update
        for _ in range(self.K_epochs):
            # Create mini-batches
            for idx in range(0, len(observations), self.batch_size):
                batch_indices = slice(idx, idx + self.batch_size)
                batch_observations = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_action_log_probs = old_action_log_probs[batch_indices]
                batch_state_values = state_values[batch_indices]
                batch_action_masks = action_masks[batch_indices]

                # Evaluate current policy
                logits, state_values_pred = self.policy_network(batch_observations)
                state_values_pred = state_values_pred.squeeze(-1)

                # Compute action probabilities
                masked_logits = logits + (batch_action_masks - 1) * 1e10
                action_probs = F.softmax(masked_logits, dim=-1)
                dist = Categorical(action_probs)

                # Compute entropy
                entropy = dist.entropy().mean()

                # Compute new action log probs
                valid_actions_mask = batch_actions != -1
                new_action_log_probs = torch.zeros_like(batch_old_action_log_probs)
                if valid_actions_mask.any():
                    new_action_log_probs[valid_actions_mask] = dist.log_prob(
                        batch_actions[valid_actions_mask]
                    )

                    # Compute ratio
                    ratios = torch.exp(
                        new_action_log_probs[valid_actions_mask]
                        - batch_old_action_log_probs[valid_actions_mask]
                    )

                    # Compute surrogate loss
                    surr1 = ratios * batch_advantages[valid_actions_mask]
                    surr2 = (
                        torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                        * batch_advantages[valid_actions_mask]
                    )
                    actor_loss = -torch.min(surr1, surr2).mean()
                else:
                    actor_loss = torch.tensor(0.0).to(self.device)

                # Critic loss (value function loss)
                critic_loss = F.mse_loss(state_values_pred, batch_state_values)

                # Total loss
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def update_entropy_coef(self):
        progress = min(1.0, self.total_episodes / self.entropy_anneal_episodes)
        self.entropy_coef = self.entropy_coef_start - progress * (
            self.entropy_coef_start - self.entropy_coef_end
        )
