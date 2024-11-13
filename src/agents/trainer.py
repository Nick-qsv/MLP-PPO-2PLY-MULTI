import torch
import torch.nn as nn
import torch.nn.functional as F
import pynvml
import time
import gc
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
                if t != 0:
                    Y_t.backward(
                        retain_graph=True
                    )  # Retain graph for subsequent backward passes
                else:
                    Y_t.backward()  # Do not retain on the last backward pass

                # Collect gradients
                gradients = {
                    name: (
                        param.grad.clone()
                        if param.grad is not None
                        else torch.zeros_like(param)
                    )
                    for name, param in zip(self.param_names, self.param_list)
                }

                # Update eligibility traces and detach
                for name in eligibility_traces:
                    eligibility_traces[name] = (
                        self.lamda * self.gamma * eligibility_traces[name]
                        + gradients[name]
                    ).detach()

                # Compute Δw[t] = α δ[t] e[t] and detach
                delta_t = delta_values[t]
                for name in self.delta_w_batch:
                    delta_w_t = self.alpha * delta_t * eligibility_traces[name]
                    self.delta_w_batch[name] += delta_w_t.detach()

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

    def update_profiling(self, episodes):
        if len(episodes) != self.batch_episode_size:
            raise ValueError(
                f"Expected {self.batch_episode_size} episodes, but got {len(episodes)}."
            )

        # Initialize CUDA events for profiling the loop
        loop_events = {
            "loop_start": torch.cuda.Event(enable_timing=True),
            "loop_end": torch.cuda.Event(enable_timing=True),
        }

        # Initialize accumulators for per-episode profiling
        accumulators = {
            "collect_observations_rewards": 0.0,
            "forward_pass": 0.0,
            "compute_td_errors": 0.0,
            "process_time_steps": 0.0,
            "process_zero_grad": 0.0,
            "process_backward": 0.0,
            "process_collect_gradients": 0.0,
            "process_update_eligibility_traces": 0.0,
            "process_compute_delta_w_t": 0.0,
            "process_update_delta_w_batch": 0.0,
            "process_zero_grad_again": 0.0,
            "episode_total": 0.0,
        }

        # Initialize maximum GPU utilization tracker
        max_gpu_util = 0
        max_mem_used = 0

        num_episodes = len(episodes)

        # Start profiling total loop time
        loop_events["loop_start"].record()

        # Begin loop over episodes
        for episode_idx, episode in enumerate(episodes):
            # Start per-episode timing
            episode_start = torch.cuda.Event(enable_timing=True)
            episode_end = torch.cuda.Event(enable_timing=True)
            episode_start.record()

            # --- Block: Collect observations and rewards ---
            collect_start = torch.cuda.Event(enable_timing=True)
            collect_end = torch.cuda.Event(enable_timing=True)
            collect_start.record()

            experiences = episode.experiences
            eligibility_traces = self.eligibility_traces_pool[episode_idx]

            observations = []
            rewards = []
            for experience in experiences:
                x_t = experience.observation  # Tensors are already on the GPU
                observations.append(x_t)
                rewards.append(experience.reward)

            observations = torch.stack(observations)  # Shape: (seq_len, input_size)
            rewards = torch.stack(rewards)  # Shape: (seq_len,)

            collect_end.record()
            torch.cuda.synchronize()
            elapsed_collect = collect_start.elapsed_time(collect_end)
            accumulators["collect_observations_rewards"] += elapsed_collect

            # --- Block: Forward pass ---
            forward_start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            forward_start.record()

            Y_values = self.policy_network(observations)  # Shape: (seq_len,)

            forward_end.record()
            torch.cuda.synchronize()
            elapsed_forward = forward_start.elapsed_time(forward_end)
            accumulators["forward_pass"] += elapsed_forward

            # --- Block: Compute TD errors ---
            td_start = torch.cuda.Event(enable_timing=True)
            td_end = torch.cuda.Event(enable_timing=True)
            td_start.record()

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

            td_end.record()
            torch.cuda.synchronize()
            elapsed_td = td_start.elapsed_time(td_end)
            accumulators["compute_td_errors"] += elapsed_td

            # --- Block: Process time steps in reverse ---
            process_start = torch.cuda.Event(enable_timing=True)
            process_end = torch.cuda.Event(enable_timing=True)
            process_start.record()

            for t in reversed(range(seq_len)):
                # --- Sub-Block: Zero Gradients ---
                zero_grad_start = torch.cuda.Event(enable_timing=True)
                zero_grad_end = torch.cuda.Event(enable_timing=True)
                zero_grad_start.record()

                self.policy_network.zero_grad()

                zero_grad_end.record()
                torch.cuda.synchronize()
                elapsed_zero_grad = zero_grad_start.elapsed_time(zero_grad_end)
                accumulators["process_zero_grad"] += elapsed_zero_grad

                # --- Sub-Block: Backward Pass ---
                backward_start = torch.cuda.Event(enable_timing=True)
                backward_end = torch.cuda.Event(enable_timing=True)
                backward_start.record()

                Y_t = Y_values[t]
                if t != 0:
                    Y_t.backward(retain_graph=True)
                else:
                    Y_t.backward()

                backward_end.record()
                torch.cuda.synchronize()
                elapsed_backward = backward_start.elapsed_time(backward_end)
                accumulators["process_backward"] += elapsed_backward

                # --- Sub-Block: Collect Gradients ---
                collect_grad_start = torch.cuda.Event(enable_timing=True)
                collect_grad_end = torch.cuda.Event(enable_timing=True)
                collect_grad_start.record()

                gradients = {
                    name: (
                        param.grad.clone()
                        if param.grad is not None
                        else torch.zeros_like(param)
                    )
                    for name, param in zip(self.param_names, self.param_list)
                }

                collect_grad_end.record()
                torch.cuda.synchronize()
                elapsed_collect_grad = collect_grad_start.elapsed_time(collect_grad_end)
                accumulators["process_collect_gradients"] += elapsed_collect_grad

                # --- Sub-Block: Update Eligibility Traces ---
                update_elig_start = torch.cuda.Event(enable_timing=True)
                update_elig_end = torch.cuda.Event(enable_timing=True)
                update_elig_start.record()

                for name in eligibility_traces:
                    eligibility_traces[name] = (
                        self.lamda * self.gamma * eligibility_traces[name]
                        + gradients[name]
                    ).detach()

                update_elig_end.record()
                torch.cuda.synchronize()
                elapsed_update_elig = update_elig_start.elapsed_time(update_elig_end)
                accumulators["process_update_eligibility_traces"] += elapsed_update_elig

                # --- Sub-Block: Compute Delta Weights ---
                compute_delta_w_start = torch.cuda.Event(enable_timing=True)
                compute_delta_w_end = torch.cuda.Event(enable_timing=True)
                compute_delta_w_start.record()

                delta_t = delta_values[t]
                for name in self.delta_w_batch:
                    delta_w_t = self.alpha * delta_t * eligibility_traces[name]
                    self.delta_w_batch[name] += delta_w_t.detach()

                compute_delta_w_end.record()
                torch.cuda.synchronize()
                elapsed_compute_delta_w = compute_delta_w_start.elapsed_time(
                    compute_delta_w_end
                )
                accumulators["process_compute_delta_w_t"] += elapsed_compute_delta_w

                # --- Sub-Block: Update Delta Weight Batch ---
                update_delta_w_batch_start = torch.cuda.Event(enable_timing=True)
                update_delta_w_batch_end = torch.cuda.Event(enable_timing=True)
                update_delta_w_batch_start.record()

                # Note: The actual update was already done in the previous sub-block
                # If there's additional processing, it can be added here.

                update_delta_w_batch_end.record()
                torch.cuda.synchronize()
                elapsed_update_delta_w_batch = update_delta_w_batch_start.elapsed_time(
                    update_delta_w_batch_end
                )
                accumulators[
                    "process_update_delta_w_batch"
                ] += elapsed_update_delta_w_batch

                # --- Sub-Block: Zero Gradients Again ---
                zero_grad_again_start = torch.cuda.Event(enable_timing=True)
                zero_grad_again_end = torch.cuda.Event(enable_timing=True)
                zero_grad_again_start.record()

                self.policy_network.zero_grad()

                zero_grad_again_end.record()
                torch.cuda.synchronize()
                elapsed_zero_grad_again = zero_grad_again_start.elapsed_time(
                    zero_grad_again_end
                )
                accumulators["process_zero_grad_again"] += elapsed_zero_grad_again

            process_end.record()
            torch.cuda.synchronize()
            elapsed_process = process_start.elapsed_time(process_end)
            accumulators["process_time_steps"] += elapsed_process

            # --- Update maximum GPU utilization ---
            gpu_util_current = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem_info_current = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            max_gpu_util = max(max_gpu_util, gpu_util_current.gpu)
            max_mem_used = max(max_mem_used, mem_info_current.used)

            # End per-episode timing
            episode_end.record()
            torch.cuda.synchronize()
            elapsed_episode = episode_start.elapsed_time(episode_end)
            accumulators["episode_total"] += elapsed_episode

        # End profiling total loop time
        loop_events["loop_end"].record()
        torch.cuda.synchronize()
        elapsed_loop = loop_events["loop_start"].elapsed_time(loop_events["loop_end"])
        print(f"[Profile] Loop over episodes took {elapsed_loop:.2f} ms")

        # Compute average timings across episodes
        avg_collect = accumulators["collect_observations_rewards"] / num_episodes
        avg_forward = accumulators["forward_pass"] / num_episodes
        avg_td = accumulators["compute_td_errors"] / num_episodes
        avg_process = accumulators["process_time_steps"] / num_episodes
        avg_zero_grad = accumulators["process_zero_grad"] / num_episodes
        avg_backward = accumulators["process_backward"] / num_episodes
        avg_collect_grad = accumulators["process_collect_gradients"] / num_episodes
        avg_update_elig = (
            accumulators["process_update_eligibility_traces"] / num_episodes
        )
        avg_compute_delta_w = accumulators["process_compute_delta_w_t"] / num_episodes
        avg_update_delta_w_batch = (
            accumulators["process_update_delta_w_batch"] / num_episodes
        )
        avg_zero_grad_again = accumulators["process_zero_grad_again"] / num_episodes
        avg_episode_total = accumulators["episode_total"] / num_episodes

        print("\n[Profile] Average Timings per Episode:")
        print(f"  Total Episode Time: {avg_episode_total:.2f} ms")
        print(f"    Collect Observations & Rewards: {avg_collect:.2f} ms")
        print(f"    Forward Pass: {avg_forward:.2f} ms")
        print(f"    Compute TD Errors: {avg_td:.2f} ms")
        print(f"    Process Time Steps: {avg_process:.2f} ms")
        print(f"      Zero Gradients: {avg_zero_grad:.2f} ms")
        print(f"      Backward Pass: {avg_backward:.2f} ms")
        print(f"      Collect Gradients: {avg_collect_grad:.2f} ms")
        print(f"      Update Eligibility Traces: {avg_update_elig:.2f} ms")
        print(f"      Compute Delta Weights: {avg_compute_delta_w:.2f} ms")
        print(f"      Update Delta Weight Batch: {avg_update_delta_w_batch:.2f} ms")
        print(f"      Zero Gradients Again: {avg_zero_grad_again:.2f} ms")

        # Print maximum GPU utilization
        print(f"\nMax GPU Utilization during loop: {max_gpu_util}%")
        print(f"Max Memory Used during loop: {max_mem_used / (1024 ** 2):.2f} MB")

        # After loop, proceed with rest of the update function (without profiling)

        # Update the total episodes
        self.total_episodes += len(episodes)

        # Average the accumulated parameter updates over the batch of episodes
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

        # Update parameters in parameter manager
        self.parameter_manager.set_parameters(self.policy_network.state_dict())

        # Profile GPU after update (if desired)
        gpu_util_after = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        mem_info_after = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)

        # Print summary
        print(f"\nUpdate completed")
        print("GPU Usage Summary:")
        print(
            f"  After Update  - Utilization: {gpu_util_after.gpu}%, Memory Used: {mem_info_after.used / (1024 ** 2):.2f} MB"
        )
        print(f"  Max Utilization during Loop: {max_gpu_util}%")
        print(f"  Max Memory Used during Loop: {max_mem_used / (1024 ** 2):.2f} MB")
