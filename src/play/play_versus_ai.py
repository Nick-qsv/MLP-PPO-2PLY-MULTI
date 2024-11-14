import sys
import os

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Add the `src` directory to the system path
sys.path.append(os.path.join(project_root, "src"))

from environments import BackgammonEnv
from agents.policy_network import BackgammonPolicyNetwork
from play.game_renderer import render
from backgammon.types import Player
import torch


def play_game():
    done = False
    env = BackgammonEnv()
    policy_network = BackgammonPolicyNetwork()
    path = os.path.join(project_root, "src/play", "backgammon_mlp_episode_340000.pth")
    try:
        policy_network.load_state_dict(
            torch.load(path, map_location=torch.device("cpu"))
        )
        print(f"Loaded model from {path}")
    except Exception as e:
        print(f"Failed to load model from {path}: {e}")

    env.reset()
    human_player = Player.PLAYER1
    while not done:
        if env.current_player == human_player:
            render(env.board, human_player)
            print("Your turn:")
            if len(env.legal_moves) == 0:
                # No legal moves, automatically pass the turn
                print("No legal moves. Passing your turn.")
                _, _, done, _ = env.step(0)  # Pass the turn
            else:
                action = human_play_step(env)
                if action is None:
                    _, _, done, _ = env.step(0)  # Pass the turn
                else:
                    _, _, done, _ = env.step(action)
        else:
            print("Agent's turn:")
            # Display the agent's current dice roll
            print("Agent Rolled: ", env.roll_result)

            if len(env.legal_moves) == 0:
                # No legal moves for AI, pass the turn
                print("No legal moves for the agent. Passing turn.")
                _, _, done, _ = env.step(0)  # Pass the turn
            else:
                action = agent_play_step(policy_network, env)
                _, _, done, _ = env.step(action)

        if done:
            print(f"Game Over. Winner: {env.current_match_winner.name}")
            break


def human_play_step(env):
    # List legal moves
    legal_moves = env.legal_moves
    if len(legal_moves) == 0:
        print("No legal moves available. Passing turn.")
        return None

    print("You Rolled: ", env.roll_result)

    print("Available moves:")
    for i, move in enumerate(legal_moves):
        # Generate the description for each SubMove
        moves_description = ", ".join(
            f"[{'bar' if sub_move.start == -1 else sub_move.start}, "
            f"{'off' if sub_move.end == -2 else sub_move.end}, "
            f"{'*' if sub_move.hits_blot else '-'}]"
            for sub_move in move.sub_move_commands
        )

        # Format the full move string
        full_move_str = f"Full Move ({move.player.name}): {moves_description}"

        # Print the formatted move
        print(f"Move {i}: {full_move_str}")

    # Let the human player select a move
    selected_move = input(f"Select a move (0 to {len(legal_moves) - 1}): ")
    if not selected_move.isdigit():
        print("Invalid input. Please enter a number.")
        return None

    selected_move = int(selected_move)
    if selected_move < 0 or selected_move >= len(legal_moves):
        print("Invalid move selected.")
        return None

    return selected_move


def agent_play_step(policy_network, env):
    # Agent selects an action based on the current observation
    action_idx = select_highest_value_action(policy_network, env.current_board_features)

    # Retrieve the selected move for descriptive printing
    selected_move = env.legal_moves[action_idx]

    # Generate a descriptive string for the selected move
    moves_description = ", ".join(
        f"[{'bar' if sub_move.start == -1 else sub_move.start}, "
        f"{'off' if sub_move.end == -2 else sub_move.end}, "
        f"{'*' if sub_move.hits_blot else '-'}]"
        for sub_move in selected_move.sub_move_commands
    )
    full_move_str = f"Full Move ({selected_move.player.name}): {moves_description}"

    # Print the agent's selected action and its description
    print(f"Agent selected action {action_idx}: {full_move_str}")
    return action_idx


def select_highest_value_action(policy_network, x):
    with torch.no_grad():
        state_values = policy_network.forward(x)

    # Extract action state values (remaining values)
    if state_values.dim() == 0:
        action_state_values = state_values  # scalar case
    else:
        action_state_values = state_values[1:]  # usual slicing if it's 1D or higher

    # Get the index of the action with the maximum state value
    action_idx = torch.argmax(
        action_state_values
    ).item()  # Selects the highest value's index

    return action_idx


if __name__ == "__main__":
    print(sys.path)
    play_game()
