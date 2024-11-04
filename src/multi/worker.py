import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from environments import Episode, Experience, BackgammonEnv
from agents import BackgammonPolicyNetwork
from config import USE_SIGMOID, MAX_TIMESTEPS


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
            print(f"Worker {self.worker_id} added an episode to the queue.")
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

        done = False
        step_count = 0

        while not done and step_count < max_steps:
            action_mask = env.action_mask.clone()
            legal_moves = env.legal_moves

            if not legal_moves:
                # No legal moves, pass turn (handled in env)
                action = None
                action_log_prob = None
                state_value = None
                next_observation, reward, done, info = env.step(action)
                next_state_value = None
            else:
                # Use PolicyNetwork to select action
                observation_tensor = observation.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits, state_value = self.policy_network(observation_tensor)
                    logits = logits.squeeze(0)
                    state_value = state_value.squeeze(0)

                    # Properly mask invalid actions
                    masked_logits = logits.clone()
                    masked_logits[action_mask == 0] = -float("inf")

                    # Compute probabilities
                    action_probs = F.softmax(masked_logits, dim=-1)

                    # Sample action
                    m = Categorical(action_probs)
                    action = m.sample()
                    action_log_prob = m.log_prob(action)
                    # print(f"Worker {self.worker_id}: Selected action {action.item()}")

                # Take action in env
                next_observation, reward, done, info = env.step(action.item())

                # Get next state value
                with torch.no_grad():
                    next_observation_tensor = next_observation.unsqueeze(0).to(
                        self.device
                    )
                    _, next_state_value = self.policy_network(next_observation_tensor)
                    next_state_value = next_state_value.squeeze(0)

            # Create Experience
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

            # Add experience to episode
            episode.add_experience(experience)

            # Move to next observation
            observation = next_observation
            step_count += 1

        if step_count >= max_steps:
            print(f"Worker {self.worker_id}: Reached maximum steps in episode.")
        return episode


def worker_function(worker_id, parameter_manager, experience_queue):
    worker = Worker(worker_id, parameter_manager, experience_queue)
    worker.run()
