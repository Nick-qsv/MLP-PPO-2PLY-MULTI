import torch
import torch.nn as nn
import torch.nn.functional as F
import pynvml
import time
from torch.distributions import Categorical
from .policy_network import BackgammonPolicyNetwork
from config import *


class Trainer:
    def __init__(self, parameter_manager, device=None):
        self.parameter_manager = parameter_manager
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
        # Initialize NVML handle for the GPU
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU

    def update(self, episodes):
        # Profile GPU before update
        gpu_util_before = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        mem_info_before = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)

        start_time = time.time()

        # Initialize maximum GPU utilization tracker
        max_gpu_util = gpu_util_before.gpu
        max_mem_used = mem_info_before.used
        # Update the total episodes
        self.total_episodes += len(episodes)

        # Collect experiences from episodes
        observations = []
        actions = []
        action_log_probs = []
        rewards = []
        dones = []
        state_values = []
        next_state_values = []
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

            for experience in episode.experiences:
                episode_observations.append(experience.observation)
                episode_actions.append(experience.action)
                episode_action_log_probs.append(experience.action_log_prob)
                episode_rewards.append(experience.reward)
                episode_dones.append(experience.done)
                episode_state_values.append(experience.state_value)
                episode_next_state_values.append(experience.next_state_value)

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

        # Convert lists to tensors
        observations = torch.stack(observations).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        action_log_probs = torch.stack(action_log_probs).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        state_values = torch.stack(state_values).to(self.device)
        next_state_values = torch.stack(next_state_values).to(self.device)
        advantages = torch.tensor(advantages).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_action_log_probs = action_log_probs.detach()

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

                # Evaluate current policy
                logits, state_values_pred = self.policy_network(batch_observations)
                state_values_pred = state_values_pred.squeeze(-1)

                # Compute action probabilities
                action_probs = F.softmax(logits, dim=-1)
                dist = Categorical(action_probs)

                # Compute entropy
                entropy = dist.entropy().mean()

                # Compute new action log probs
                new_action_log_probs = dist.log_prob(batch_actions)

                # Compute ratio
                ratios = torch.exp(new_action_log_probs - batch_old_action_log_probs)

                # Compute surrogate loss
                surr1 = ratios * batch_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss (value function loss)
                critic_loss = F.mse_loss(state_values_pred, batch_state_values)

                # Total loss
                loss = (
                    actor_loss
                    + VALUE_LOSS_COEF * critic_loss
                    - self.entropy_coef * entropy
                )

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update maximum GPU utilization
                torch.cuda.synchronize()
                current_gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                current_mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                max_gpu_util = max(max_gpu_util, current_gpu_util.gpu)
                max_mem_used = max(max_mem_used, current_mem_info.used)

        # After update, update entropy coefficient
        self.update_entropy_coef()
        # After update, update parameters in parameter manager
        self.parameter_manager.set_parameters(self.policy_network.state_dict())
        self.parameter_manager.increment_version()

        end_time = time.time()

        # Profile GPU after update
        gpu_util_after = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        mem_info_after = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)

        # Print summary
        print(f"Update completed in {end_time - start_time:.2f} seconds")
        print("GPU Usage Summary:")
        print(
            f"  Before Update - Utilization: {gpu_util_before.gpu}%, Memory Used: {mem_info_before.used / (1024 ** 2):.2f} MB"
        )
        print(
            f"  After Update  - Utilization: {gpu_util_after.gpu}%, Memory Used: {mem_info_after.used / (1024 ** 2):.2f} MB"
        )
        print(f"  Max Utilization during Update: {max_gpu_util}%")
        print(f"  Max Memory Used during Update: {max_mem_used / (1024 ** 2):.2f} MB")

    def update_entropy_coef(self):
        progress = min(1.0, self.total_episodes / self.entropy_anneal_episodes)
        self.entropy_coef = self.entropy_coef_start - progress * (
            self.entropy_coef_start - self.entropy_coef_end
        )
