import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from environments import Episode, Experience, BackgammonEnv
from agents import BackgammonPolicyNetwork
from config import MAX_TIMESTEPS
import time
import torch.autograd.profiler as profiler


# about to change to TD from PPO
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
        self.policy_network = BackgammonPolicyNetwork()
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
        """
        Execute a single episode in the given environment using the policy network.

        Args:
            env (Environment): The environment to interact with, providing methods like `reset()` and `step(action)`, and attributes such as `num_moves`, `legal_board_features`, and `legal_moves`.
            max_steps (int, optional): Maximum number of steps to run in the episode. Defaults to `MAX_TIMESTEPS`.

        Returns:
            Episode: An `Episode` object containing the sequence of experiences gathered during the episode.

        Notes:
            - Profiles the execution time of key operations and prints profiling data after the episode.
            - Handles cases with no legal moves by taking a no-op action.
        """
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
            # Retrieve the exact number of valid legal moves
            num_moves = env.num_moves  # Number of valid legal moves

            # Handle the case where there are no legal moves
            if num_moves == 0:
                # No legal moves, take a no-op action (pass turn)
                start_timer("Env Step (No Action)")
                next_observation, reward, done, info = env.step(None)
                end_timer("Env Step (No Action)")

                # Move to next observation
                observation = next_observation
                step_count += 1
                continue  # Skip adding experience and forward pass

            # Prepare valid action states and concatenate with current observation
            resulting_states = env.legal_board_features[
                :num_moves
            ]  # Shape: (num_moves, 198)
            x = torch.cat(
                [observation.unsqueeze(0), resulting_states], dim=0
            )  # Shape: (num_moves + 1, 198)

            # Perform a forward pass using forward_combined
            start_timer("Policy Network Forward Pass on Actions")
            with torch.no_grad():
                logits, state_values = self.policy_network.forward_combined(x)
            end_timer("Policy Network Forward Pass on Actions")
            # Extract state values
            original_state_value = state_values[
                0
            ].item()  # State value for current observation
            action_state_values = state_values[
                1:
            ]  # State values for resulting states (num_moves,)
            # Sample an action stochastically
            start_timer("Sample Action Stochastically")
            action_logits = logits  # Logits correspond to actions
            action_probs = F.softmax(action_logits, dim=0)
            m = Categorical(probs=action_probs)
            action_idx = m.sample().item()
            next_state_value = action_state_values[action_idx].item()
            action_log_prob = m.log_prob(torch.tensor(action_idx)).item()
            end_timer("Sample Action Stochastically")

            # Take action in env
            start_timer("Env Step")
            next_observation, reward, done, info = env.step(action_idx)
            end_timer("Env Step")

            # Create and add Experience
            experience = Experience(
                observation=observation,
                action=action_idx,
                action_log_prob=action_log_prob,
                state_value=original_state_value,
                reward=reward,
                done=done,
                next_observation=next_observation,
                next_state_value=next_state_value,
            )

            episode.add_experience(experience)

            # Move to next observation
            observation = next_observation
            step_count += 1

        if step_count >= max_steps:
            print(f"Worker {self.worker_id}: Reached maximum steps in episode.")

        # Convert tensors to NumPy arrays before returning the episode
        episode.to_numpy()

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
