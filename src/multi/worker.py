import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from environments import Episode, Experience, BackgammonEnv
from agents import BackgammonPolicyNetwork
from config import USE_SIGMOID, MAX_TIMESTEPS
import time


class Worker:
    """
    Worker class that runs episodes of Backgammon games using a local copy of the PolicyNetwork.
    It interacts with the BackgammonEnvironment, periodically checks for parameter updates from
    the ParameterManager, and stores completed episodes in the ExperienceQueue.

    This class is designed to be scalable across multiple CPUs.
    """

    def __init__(self, worker_id, parameter_manager, experience_queue):
        """
        Initializes the Worker.

        Args:
            worker_id (int): Unique identifier for the worker.
            parameter_manager (ParameterManager): Shared ParameterManager instance.
            experience_queue (ExperienceQueue): Shared queue for storing episodes.
            device (torch.device, optional): Device to run computations on. Defaults to CPU.
        """
        self.worker_id = worker_id
        self.parameter_manager = parameter_manager
        self.experience_queue = experience_queue
        self.device = torch.device("cpu")

    def run(self):
        """
        Main loop for the worker. Runs continuously, playing episodes and checking for parameter updates.
        """
        self.policy_network = BackgammonPolicyNetwork(use_sigmoid=USE_SIGMOID)
        state_dict = self.parameter_manager.get_parameters()
        self.policy_network.load_state_dict(state_dict)
        self.current_version = self.parameter_manager.get_version()
        # Create environment
        env = BackgammonEnv(worker_id=self.worker_id, device=self.device)
        print(f"Worker {self.worker_id} starting.")
        while True:
            # Play an episode
            episode = self.play_episode(env)

            # Put the episode into the ExperienceQueue
            self.experience_queue.put(episode)
            # print(f"Worker {self.worker_id} added an episode to the queue.")
            # At the end of the episode, check for updated parameters
            new_version = self.parameter_manager.get_version()
            if new_version > self.current_version:
                # Update local PolicyNetwork parameters
                state_dict = self.parameter_manager.get_parameters()
                self.policy_network.load_state_dict(state_dict)
                self.current_version = new_version
                print(
                    f"Worker {self.worker_id}: Updated parameters to version {self.current_version}"
                )

    def play_episode(self, env, max_steps=MAX_TIMESTEPS):
        episode = Episode()
        observation = env.reset()

        # Initialize profiling dictionary
        profiling_data = {}

        def start_timer(key):
            profiling_data[key] = profiling_data.get(
                key, {"total_time": 0.0, "call_count": 0}
            )
            profiling_data[key]["start_time"] = time.perf_counter()

        def end_timer(key):
            end_time = time.perf_counter()
            profiling_data[key]["total_time"] += (
                end_time - profiling_data[key]["start_time"]
            )
            profiling_data[key]["call_count"] += 1

        done = False
        step_count = 0

        while not done and step_count < max_steps:
            # Start profiling for environment action mask retrieval
            start_timer("Retrieve Action Mask")
            action_mask = env.action_mask.clone()  # Shape: (500,)
            legal_moves = env.legal_moves
            end_timer("Retrieve Action Mask")

            # Prepare combined_states and combined_action_mask
            start_timer("Prepare Combined States and Action Mask")
            original_observation = observation.unsqueeze(0)  # Shape: (1, 198)
            if legal_moves:
                resulting_states = env.legal_board_features  # Shape: (500, 198)
                combined_states = torch.cat(
                    [original_observation, resulting_states], dim=0
                )  # Shape: (501, 198)

                # Prepare combined action mask
                original_mask = torch.tensor([0], dtype=torch.float32)
                legal_masks = action_mask.to(
                    dtype=torch.float32, device=self.device
                )  # Shape: (500,)
                combined_action_mask = torch.cat(
                    [original_mask, legal_masks], dim=0
                )  # Shape: (501,)
            else:
                # No legal moves, only the original observation
                combined_states = original_observation  # Shape: (1, 198)
                combined_action_mask = torch.tensor(
                    [0], dtype=torch.float32
                )  # Shape: (1,)
            end_timer("Prepare Combined States and Action Mask")

            # Perform a forward pass
            start_timer("Policy Network Forward Pass (Combined States)")
            with torch.no_grad():
                logits, state_values = self.policy_network(
                    combined_states, combined_action_mask
                )
            end_timer("Policy Network Forward Pass (Combined States)")

            # Split the outputs
            start_timer("Split Outputs")
            original_logit = logits[0]  # Not selectable
            original_state_value = state_values[0]
            if legal_moves:
                action_logits = logits[1:]  # Shape: (500,)
                action_state_values = state_values[1:]  # Shape: (500,)
            else:
                action_logits = torch.tensor([])
                action_state_values = torch.tensor([])
            end_timer("Split Outputs")

            if legal_moves:
                # Sample an action stochastically
                start_timer("Sample Action Stochastically")
                if action_logits.numel() > 0 and action_mask.sum() > 0:
                    # Apply masking to logits
                    masked_logits = action_logits.clone()
                    masked_logits[action_mask == 0] = -float(
                        "inf"
                    )  # Mask invalid actions

                    # Convert logits to probabilities
                    action_probs = torch.exp(masked_logits)

                    # Create a Categorical distribution
                    m = torch.distributions.Categorical(probs=action_probs)

                    # Sample an action
                    action_idx = m.sample().item()  # Index in action_logits

                    # Get the next state value
                    next_state_value = action_state_values[action_idx].item()

                    # Get the log probability of the sampled action
                    action_log_prob = (
                        masked_logits[action_idx].item()
                        - torch.logsumexp(masked_logits, dim=0).item()
                    )
                else:
                    # Handle the case where no valid actions are found
                    print(
                        f"Worker {self.worker_id}: No valid actions found despite having legal moves."
                    )
                    action = None
                    action_log_prob = None
                    next_state_value = None
                end_timer("Sample Action Stochastically")

                # Take action in env
                start_timer("Env Step")
                next_observation, reward, done, info = env.step(action_idx)
                end_timer("Env Step")
            else:
                # No legal moves, pass turn
                action = None
                action_log_prob = None
                next_state_value = None
                # Take action in env (pass)
                start_timer("Env Step (No Action)")
                next_observation, reward, done, info = env.step(action)
                end_timer("Env Step (No Action)")

            # Create Experience
            start_timer("Create Experience")
            experience = Experience(
                observation=observation,
                action_mask=action_mask,
                action=action_idx if legal_moves else None,
                action_log_prob=action_log_prob,
                state_value=(
                    original_state_value.item()
                    if original_state_value is not None
                    else None
                ),
                reward=reward,
                done=done,
                next_observation=next_observation,
                next_state_value=next_state_value,
            )
            end_timer("Create Experience")

            # Add experience to episode
            start_timer("Add Experience to Episode")
            episode.add_experience(experience)
            end_timer("Add Experience to Episode")

            # Move to next observation
            observation = next_observation
            step_count += 1

        if step_count >= max_steps:
            print(f"Worker {self.worker_id}: Reached maximum steps in episode.")

        # Convert tensors to NumPy arrays before returning the episode
        start_timer("Convert Episode to NumPy")
        episode.to_numpy()
        end_timer("Convert Episode to NumPy")

        # Print profiling data
        print(f"\nWorker {self.worker_id} profiling data for this episode:")
        for func_name, data in profiling_data.items():
            total_time = data["total_time"]
            call_count = data["call_count"]
            avg_time = total_time / call_count if call_count else 0
            print(
                f"{func_name}: Total Time = {total_time:.6f}s, Calls = {call_count}, Average Time = {avg_time:.6f}s"
            )

        return episode


def worker_function(worker_id, parameter_manager, experience_queue):
    worker = Worker(worker_id, parameter_manager, experience_queue)
    worker.run()
