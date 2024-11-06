import time
from functools import wraps

profiling_data = {}


def profile(func):
    @wraps(func)
    def wrapper_profile(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        func_name = func.__name__
        if func_name not in profiling_data:
            profiling_data[func_name] = {"total_time": 0.0, "call_count": 0}
        profiling_data[func_name]["total_time"] += elapsed
        profiling_data[func_name]["call_count"] += 1
        return result

    return wrapper_profile

    # def play_episode_old(self, env, max_steps=MAX_TIMESTEPS):
    #     episode = Episode()
    #     observation = env.reset()

    #     done = False
    #     step_count = 0

    #     while not done and step_count < max_steps:
    #         action_mask = env.action_mask.clone()
    #         legal_moves = env.legal_moves

    #         if not legal_moves:
    #             # No legal moves, pass turn (handled in env)
    #             action = None
    #             action_log_prob = None
    #             state_value = None
    #             next_observation, reward, done, info = env.step(action)
    #             next_state_value = None
    #         else:
    #             # Use PolicyNetwork to select action
    #             observation_tensor = observation.unsqueeze(0).to(self.device)

    #             with torch.no_grad():
    #                 logits, state_value = self.policy_network(observation_tensor)
    #                 logits = logits.squeeze(0)  # Shape: (action_size,)
    #                 state_value = state_value.squeeze(0)

    #             # Properly mask invalid actions
    #             masked_logits = logits.clone()
    #             masked_logits[action_mask == 0] = -float("inf")

    #             # Compute probabilities
    #             action_probs = F.softmax(masked_logits, dim=-1)

    #             # Get indices of legal actions
    #             legal_action_indices = torch.nonzero(action_mask).squeeze(-1)

    #             # Get legal next observations (board features)
    #             num_legal_moves = len(legal_moves)
    #             legal_board_features = env.legal_board_features[:num_legal_moves].to(
    #                 self.device
    #             )

    #             # Pass legal next observations through policy network to get their state values
    #             with torch.no_grad():
    #                 _, next_state_values = self.policy_network(legal_board_features)
    #                 next_state_values = next_state_values.squeeze(
    #                     -1
    #                 )  # Shape: (num_legal_moves,)

    #             # Select the action that leads to the next state with the highest state value
    #             best_legal_action_idx = torch.argmax(next_state_values)

    #             # Get the corresponding action index in the action space
    #             action = legal_action_indices[best_legal_action_idx]

    #             # Get the action_log_prob of the selected action
    #             action_log_prob = torch.log(action_probs[action])

    #             # Take action in env
    #             next_observation, reward, done, info = env.step(action.item())

    #             # Get next state value
    #             with torch.no_grad():
    #                 next_observation_tensor = next_observation.unsqueeze(0).to(
    #                     self.device
    #                 )
    #                 _, next_state_value = self.policy_network(next_observation_tensor)
    #                 next_state_value = next_state_value.squeeze(0)

    #         # Create Experience
    #         experience = Experience(
    #             observation=observation,
    #             action_mask=action_mask,
    #             action=action,
    #             action_log_prob=action_log_prob,
    #             state_value=state_value,
    #             reward=reward,
    #             done=done,
    #             next_observation=next_observation,
    #             next_state_value=next_state_value,
    #         )

    #         # Add experience to episode
    #         episode.add_experience(experience)

    #         # Move to next observation
    #         observation = next_observation
    #         step_count += 1

    #     if step_count >= max_steps:
    #         print(f"Worker {self.worker_id}: Reached maximum steps in episode.")

    #     # Convert tensors to NumPy arrays before returning the episode
    #     episode.to_numpy()

    #     # Print profiling data
    #     print(f"Worker {self.worker_id} profiling data for this episode:")
    #     for func_name, data in env.profiling_data.items():
    #         total_time = data["total_time"]
    #         call_count = data["call_count"]
    #         avg_time = total_time / call_count if call_count else 0
    #         print(
    #             f"{func_name}: Total Time = {total_time:.6f}s, Calls = {call_count}, Average Time = {avg_time:.6f}s"
    #         )

    #     # Reset profiling data for the next episode
    #     env.profiling_data = {}

    #     return episode
