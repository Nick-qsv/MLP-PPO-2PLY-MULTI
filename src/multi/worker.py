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
        self.policy_network = BackgammonPolicyNetwork(use_sigmoid=USE_SIGMOID).to(
            self.device
        )
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
            action_mask = env.action_mask.clone()
            legal_moves = env.legal_moves
            end_timer("Retrieve Action Mask")

            if not legal_moves:
                # No legal moves, pass turn (handled in env)
                start_timer("Env Step (No Action)")
                action = None
                action_log_prob = None
                state_value = None
                next_state_value = None
                next_observation, reward, done, info = env.step(action)
                end_timer("Env Step (No Action)")
            else:
                # Use PolicyNetwork to select action
                start_timer("Policy Network Forward Pass (Observation)")
                observation_tensor = observation.unsqueeze(0)

                with torch.no_grad():
                    logits, state_value = self.policy_network(observation_tensor)
                    logits = logits.squeeze(0)  # Shape: (action_size,)
                    state_value = state_value.squeeze(0)
                end_timer("Policy Network Forward Pass (Observation)")

                # Properly mask invalid actions
                start_timer("Mask Invalid Actions")
                masked_logits = logits.clone()
                masked_logits[action_mask == 0] = -float("inf")
                end_timer("Mask Invalid Actions")

                # Compute probabilities
                start_timer("Compute Softmax")
                action_probs = F.softmax(masked_logits, dim=-1)
                end_timer("Compute Softmax")

                # Get indices of legal actions
                start_timer("Get Legal Action Indices")
                legal_action_indices = torch.nonzero(action_mask).squeeze(-1)
                end_timer("Get Legal Action Indices")

                # Get legal next observations (board features)
                start_timer("Prepare Legal Board Features")
                num_legal_moves = len(legal_moves)
                legal_board_features = env.legal_board_features[:num_legal_moves]
                end_timer("Prepare Legal Board Features")

                # Pass legal next observations (board features) through policy network to get their state values
                start_timer("Policy Network Forward Pass (Legal Moves)")
                with torch.no_grad():
                    _, next_state_values = self.policy_network(legal_board_features)
                    next_state_values = next_state_values.view(
                        -1
                    )  # Shape: (num_legal_moves,)
                end_timer("Policy Network Forward Pass (Legal Moves)")

                # Select the action that leads to the next state with the highest state value
                start_timer("Select Best Action")
                if legal_action_indices.numel() > 0:
                    best_legal_action_idx = torch.argmax(next_state_values)
                    # Get the corresponding action index in the action space
                    action = legal_action_indices[best_legal_action_idx]
                    # Get the next_state_value corresponding to the chosen action
                    next_state_value = next_state_values[best_legal_action_idx]
                else:
                    # Handle the case where no valid actions are found
                    print(
                        f"Worker {self.worker_id}: No valid actions found despite having legal moves."
                    )
                    action = None
                    next_state_value = None
                end_timer("Select Best Action")

                if action is not None:
                    # Get the action_log_prob of the selected action
                    start_timer("Compute Action Log Prob")
                    action_log_prob = torch.log(action_probs[action])
                    end_timer("Compute Action Log Prob")

                    # Take action in env
                    start_timer("Env Step (With Action)")
                    next_observation, reward, done, info = env.step(action.item())
                    end_timer("Env Step (With Action)")

                    # No need for a third forward pass; next_state_value is already obtained
                else:
                    # Handle the scenario where no action is taken
                    action_log_prob = None
                    state_value = None
                    next_state_value = None
                    # Assuming env.step(None) handles passing the turn
                    start_timer("Env Step (No Action)")
                    next_observation, reward, done, info = env.step(action)
                    end_timer("Env Step (No Action)")

            # Create Experience
            start_timer("Create Experience")
            experience = Experience(
                observation=observation,
                action_mask=action_mask,
                action=action,
                action_log_prob=action_log_prob,
                state_value=state_value,
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
