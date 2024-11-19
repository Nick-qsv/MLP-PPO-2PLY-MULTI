import torch
import torch.nn.functional as F
import pynvml
import time
from .policy_network import BackgammonPolicyNetwork
from .logger import S3Logger
from config import *


class Trainer:
    def __init__(
        self, parameter_manager, device=None, s3_bucket_name=None, s3_log_prefix="logs/"
    ):
        self.parameter_manager = parameter_manager
        self.device = device if device is not None else torch.device("cpu")

        # Initialize policy network
        self.policy_network = BackgammonPolicyNetwork().to(self.device)

        # Load initial parameters from parameter manager
        state_dict = self.parameter_manager.get_parameters()
        # Move state_dict tensors to self.device
        state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
        self.policy_network.load_state_dict(state_dict)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=LEARNING_RATE
        )

        # Hyperparameters
        self.gamma = torch.tensor(GAMMA, device=self.device)

        self.total_episodes = 0
        self.grad_clip = GRAD_CLIP_THRESHOLD
        self.lr_decay = LR_DECAY
        self.lr_decay_steps = LR_DECAY_STEPS
        self.batch_episode_size = MIN_EPISODES_TO_TRAIN

        # Initialize NVML handle for the GPU
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU

        # Initialize S3Logger
        self.logger = S3Logger(
            s3_bucket_name=s3_bucket_name, s3_log_prefix=s3_log_prefix
        )

    def update(self, episodes):
        if len(episodes) != self.batch_episode_size:
            raise ValueError(
                f"Expected {self.batch_episode_size} episodes, but got {len(episodes)}."
            )

        # Initialize win counts
        win_counts = {"regular": 0, "gammon": 0, "backgammon": 0}

        # Other accumulators for metrics
        total_loss = 0.0
        total_td_error = 0.0
        total_grad_norm = 0.0
        total_predicted_value = 0.0
        total_reward = 0.0
        total_episode_length = 0

        for episode in episodes:
            experiences = episode.experiences

            # Collect observations and rewards
            observations = []
            rewards = []
            for experience in experiences:
                x_t = experience.observation
                observations.append(x_t)
                rewards.append(experience.reward)

            # Stack observations and rewards
            observations = torch.stack(observations)
            rewards = torch.stack(rewards).squeeze()

            # Forward pass
            Y_values = self.policy_network(observations).squeeze()

            # Compute target values
            seq_len = len(experiences)
            target_values = rewards.clone()

            if seq_len > 1:
                target_values[:-1] += self.gamma * Y_values[1:].detach()

            # Compute loss
            loss = F.mse_loss(Y_values, target_values)

            # Zero gradients and backpropagate
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.policy_network.parameters(), self.grad_clip
                )

            # Compute gradient norms
            grad_norm = 0.0
            for p in self.policy_network.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm**0.5
            total_grad_norm += grad_norm

            # Update parameters
            self.optimizer.step()

            # Compute TD error
            TD_error = target_values - Y_values
            mean_td_error = TD_error.abs().mean().item()
            total_td_error += mean_td_error

            # Accumulate other metrics
            total_loss += loss.item()
            total_predicted_value += Y_values.mean().item()
            total_reward += rewards.sum().item()
            total_episode_length += seq_len

            # Update win counts
            if episode.win_type in win_counts:
                win_counts[episode.win_type] += 1

        # Compute average metrics
        batch_size = len(episodes)
        average_loss = total_loss / batch_size
        average_td_error = total_td_error / batch_size
        average_grad_norm = total_grad_norm / batch_size
        average_predicted_value = total_predicted_value / batch_size
        average_reward = total_reward / batch_size
        average_episode_length = total_episode_length / batch_size

        # Update parameters in parameter manager
        self.parameter_manager.set_parameters(self.policy_network.state_dict())

        # Log metrics using S3Logger
        global_step = self.total_episodes

        self.logger.add_scalar("Loss/Training Loss", average_loss, global_step)
        self.logger.add_scalar("TD Error/Mean TD Error", average_td_error, global_step)
        self.logger.add_scalar(
            "Gradients/Gradient Norm", average_grad_norm, global_step
        )
        self.logger.add_scalar(
            "Values/Average Predicted Value", average_predicted_value, global_step
        )
        self.logger.add_scalar(
            "Rewards/Average Reward per Episode", average_reward, global_step
        )
        self.logger.add_scalar(
            "Episode/Average Episode Length", average_episode_length, global_step
        )
        self.logger.add_scalar("Wins/Regular", win_counts["regular"], global_step)
        self.logger.add_scalar("Wins/Gammon", win_counts["gammon"], global_step)
        self.logger.add_scalar("Wins/Backgammon", win_counts["backgammon"], global_step)

        # Log weight and bias histograms
        for name, param in self.policy_network.named_parameters():
            self.logger.add_histogram(
                f"Parameters/{name}", param.detach().cpu().numpy(), global_step
            )

        # Flush the logger to ensure metrics are written
        self.logger.writer.flush()
