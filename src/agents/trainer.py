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
        self.alpha = LEARNING_RATE
        self.total_episodes = 0
        self.grad_clip = GRAD_CLIP_THRESHOLD
        self.lr_decay = LR_DECAY
        self.lr_decay_steps = LR_DECAY_STEPS
        self.batch_episode_size = MIN_EPISODES_TO_TRAIN

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

        # Collect parameters and names
        param_list = [param for _, param in self.policy_network.named_parameters()]
        param_names = [name for name, _ in self.policy_network.named_parameters()]

        # Initialize accumulators for parameter updates
        delta_w_batch = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in zip(param_names, param_list)
        }

        # Loop over each episode
        for episode in episodes:
            experiences = episode.experiences

            # Initialize eligibility traces for parameters
            eligibility_traces = {
                name: torch.zeros_like(param, device=self.device)
                for name, param in zip(param_names, param_list)
            }

            # Collect all observations in the episode
            observations = []
            rewards = []
            for experience in experiences:
                x_t = experience.observation.to(self.device)
                observations.append(x_t)
                rewards.append(experience.reward.to(self.device))

            # Stack observations into a tensor
            observations = torch.stack(observations)  # Shape: (seq_len, input_size)
            rewards = torch.stack(rewards)  # Shape: (seq_len,)

            # Forward pass for all time steps in the episode
            Y_values = self.policy_network(observations)  # Shape: (seq_len,)

            # Compute TD errors δ[t]
            delta_values = []
            for t in range(len(experiences)):
                if t < len(experiences) - 1:
                    delta_t = Y_values[t + 1].detach() - Y_values[t]
                else:
                    # Terminal state
                    delta_t = rewards[t] - Y_values[t]
                delta_values.append(delta_t)

            # Process each time step to update eligibility traces and accumulate parameter updates
            for t in range(len(experiences)):
                # Zero gradients to prevent accumulation from previous iterations
                self.policy_network.zero_grad()

                # Compute gradient ∇Y[t] w.r.t parameters
                Y_t = Y_values[t]
                # Since Y_t is a scalar, gradients will be computed correctly
                Y_t.backward(retain_graph=True)

                # Collect gradients
                gradients = {
                    name: (
                        param.grad.clone()
                        if param.grad is not None
                        else torch.zeros_like(param)
                    )
                    for name, param in zip(param_names, param_list)
                }

                # Update eligibility traces
                for name in eligibility_traces:
                    eligibility_traces[name] = (
                        self.lamda * self.gamma * eligibility_traces[name]
                        + gradients[name]
                    )

                # Compute Δw[t] = α δ[t] e[t]
                delta_t = delta_values[t]
                for name in delta_w_batch:
                    delta_w_t = self.alpha * delta_t * eligibility_traces[name]
                    delta_w_batch[name] += delta_w_t

                # Clear gradients after use
                self.policy_network.zero_grad()

                # Update maximum GPU utilization
                gpu_util_current = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem_info_current = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                max_gpu_util = max(max_gpu_util, gpu_util_current.gpu)
                max_mem_used = max(max_mem_used, mem_info_current.used)

        # Apply the accumulated parameter updates
        with torch.no_grad():
            for name, param in zip(param_names, param_list):
                # Optionally, apply gradient clipping
                if self.grad_clip is not None:
                    delta_w = torch.clamp(
                        delta_w_batch[name], -self.grad_clip, self.grad_clip
                    )
                else:
                    delta_w = delta_w_batch[name]
                param.add_(delta_w)

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
