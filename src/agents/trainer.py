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
        # Move state_dict tensors to self.device
        state_dict = {k: v.to(self.device) for k, v in state_dict.items()}
        self.policy_network.load_state_dict(state_dict)
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=LEARNING_RATE
        )

        # Hyperparameters
        self.gamma = torch.tensor(GAMMA, device=self.device)
        self.lamda = torch.tensor(LAMBDA, device=self.device)
        self.alpha = torch.tensor(LEARNING_RATE, device=self.device)

        self.total_episodes = 0
        self.grad_clip = GRAD_CLIP_THRESHOLD
        self.lr_decay = LR_DECAY
        self.lr_decay_steps = LR_DECAY_STEPS
        self.batch_episode_size = MIN_EPISODES_TO_TRAIN

        # Initialize NVML handle for the GPU
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU

        # Collect parameter names and list
        self.param_names = [name for name, _ in self.policy_network.named_parameters()]
        self.param_list = [param for _, param in self.policy_network.named_parameters()]

        # Initialize tensor pools
        self._initialize_tensor_pools()

    def _initialize_tensor_pools(self):
        """
        Preallocate tensors for delta_w_batch and eligibility_traces.
        """
        # Preallocate delta_w_batch (single set)
        self.delta_w_batch = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in zip(self.param_names, self.param_list)
        }

        # Preallocate a pool of eligibility_traces for 100 episodes
        self.eligibility_traces_pool = [
            {
                name: torch.zeros_like(param, device=self.device)
                for name, param in zip(self.param_names, self.param_list)
            }
            for _ in range(self.batch_episode_size)
        ]

    def _reset_tensor_pools(self):
        """
        Reset (zero) all tensors in delta_w_batch and eligibility_traces_pool using list comprehensions.
        """
        # Reset delta_w_batch
        [tensor.zero_() for tensor in self.delta_w_batch.values()]

        # Reset all eligibility_traces
        [
            tensor.zero_()
            for eligibility_traces in self.eligibility_traces_pool
            for tensor in eligibility_traces.values()
        ]

    def update(self, episodes):
        if len(episodes) != self.batch_episode_size:
            raise ValueError(
                f"Expected {self.batch_episode_size} episodes, but got {len(episodes)}."
            )

        # Reset tensor pools before starting the update
        self._reset_tensor_pools()

        # Profile GPU before update
        gpu_util_before = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        mem_info_before = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)

        start_time = time.time()

        # Initialize maximum GPU utilization tracker
        max_gpu_util = gpu_util_before.gpu
        max_mem_used = mem_info_before.used

        # Inside your update function, before the problematic operation

        # Update the total episodes
        self.total_episodes += len(episodes)

        # Loop over each episode
        for episode_idx, episode in enumerate(episodes):
            experiences = episode.experiences

            # Retrieve the preallocated eligibility_traces for this episode
            eligibility_traces = self.eligibility_traces_pool[episode_idx]

            # Collect all observations and rewards in the episode
            observations = []
            rewards = []
            for experience in experiences:
                x_t = experience.observation  # Tensors are already on the GPU
                observations.append(x_t)
                rewards.append(experience.reward)

            # Stack observations and rewards into tensors
            observations = torch.stack(observations)  # Shape: (seq_len, input_size)
            rewards = torch.stack(rewards)  # Shape: (seq_len,)

            # Forward pass for all time steps in the episode
            Y_values = self.policy_network(observations)  # Shape: (seq_len,)

            # Compute TD errors δ[t]
            delta_values = []
            seq_len = len(experiences)
            for t in range(seq_len):
                if t < seq_len - 1:
                    delta_t = (
                        rewards[t] + self.gamma * Y_values[t + 1].detach() - Y_values[t]
                    )
                else:
                    # Terminal state
                    delta_t = rewards[t] - Y_values[t]
                delta_values.append(delta_t)

            # Process each time step in reverse to correctly update eligibility traces
            for t in reversed(range(seq_len)):
                # Zero gradients to prevent accumulation from previous iterations
                self.policy_network.zero_grad()

                # Compute gradient ∇Y[t] w.r.t parameters
                Y_t = Y_values[t]
                Y_t.backward(retain_graph=True)

                # Collect gradients
                gradients = {
                    name: (
                        param.grad.clone()
                        if param.grad is not None
                        else torch.zeros_like(param)
                    )
                    for name, param in zip(self.param_names, self.param_list)
                }

                # Update eligibility traces
                for name in eligibility_traces:
                    eligibility_traces[name] = (
                        self.lamda * self.gamma * eligibility_traces[name]
                        + gradients[name]
                    )

                # Compute Δw[t] = α δ[t] e[t]
                delta_t = delta_values[t]
                for name in self.delta_w_batch:
                    delta_w_t = self.alpha * delta_t * eligibility_traces[name]
                    self.delta_w_batch[name] += delta_w_t

                # Clear gradients after use
                self.policy_network.zero_grad()

            # Update maximum GPU utilization
            gpu_util_current = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem_info_current = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            max_gpu_util = max(max_gpu_util, gpu_util_current.gpu)
            max_mem_used = max(max_mem_used, mem_info_current.used)

        # Average the accumulated parameter updates over the batch of episodes
        num_episodes = len(episodes)
        with torch.no_grad():
            for name, param in zip(self.param_names, self.param_list):
                # Average the parameter updates
                delta_w_avg = self.delta_w_batch[name] / num_episodes

                # Optionally, apply gradient clipping
                if self.grad_clip is not None:
                    delta_w_avg = torch.clamp(
                        delta_w_avg, -self.grad_clip, self.grad_clip
                    )

                # Update parameters
                param.add_(delta_w_avg)

        # Update learning rate if decay is set
        if self.lr_decay:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.lr_decay ** (
                    self.total_episodes / self.lr_decay_steps
                )

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
