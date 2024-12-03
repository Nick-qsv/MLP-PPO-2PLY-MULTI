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
    path = os.path.join(
        project_root, "src/play", "backgammon_256_standard_episode_2100000.pth"
    )
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


def play_game_human_select_agent_moves():
    """
    Similar to play_game(), but allows the human to select the agent's moves.
    During the agent's turn, it displays each possible move along with its state value,
    and the human selects which move the agent should take.
    """
    done = False
    env = BackgammonEnv()
    policy_network = BackgammonPolicyNetwork()
    path = os.path.join(
        project_root, "src/play", "backgammon_80n_HT_episode_240000.pth"
    )
    try:
        policy_network.load_state_dict(
            torch.load(path, map_location=torch.device("cpu"))
        )
        print(f"Loaded model from {path}")
    except Exception as e:
        print(f"Failed to load model from {path}: {e}")

    env.reset()
    human_player = Player.PLAYER1
    ai_player = Player.PLAYER2
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
            render(env.board, ai_player)
            print("Agent's turn:")
            # Display the agent's current dice roll
            print("Agent Rolled: ", env.roll_result)

            if len(env.legal_moves) == 0:
                # No legal moves for AI, pass the turn
                print("No legal moves for the agent. Passing turn.")
                _, _, done, _ = env.step(0)  # Pass the turn
            else:
                # Display all possible agent moves with their state values
                state_values = get_state_values(policy_network, env)
                agent_move = human_select_agent_move(env, state_values)
                if agent_move is None:
                    _, _, done, _ = env.step(0)  # Pass the turn
                else:
                    _, _, done, _ = env.step(agent_move)

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
    state_values = env.legal_board_features
    num_moves = env.num_moves
    action_idx = select_highest_value_action(policy_network, state_values[:num_moves])

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

    # Get the index of the action with the maximum state value
    action_idx = torch.argmax(state_values).item()  # Selects the highest value's index

    return action_idx


def get_state_values(policy_network, env):
    """
    Retrieves the state values for all legal moves in the current environment.
    """
    state_values = env.legal_board_features
    num_moves = env.num_moves
    x = state_values[:num_moves]
    print("----- Debug: Input to Policy Network -----")
    print(f"Type of x: {type(x)}")
    if isinstance(x, torch.Tensor):
        print(f"Shape of x: {x.shape}")
        print(f"Data of x: {x}")
    else:
        print(f"x: {x}")
    print("-----------------------------------------")
    with torch.no_grad():
        state_values = policy_network.forward(x)

    # Assuming state_values correspond to the legal moves in order
    # If not, adjust this function accordingly
    return state_values


def human_select_agent_move(env, state_values):
    """
    Allows the human to select the agent's move by displaying each move with its state value.
    """
    legal_moves = env.legal_moves
    print("Agent's Available Moves with State Values:")
    for i, move in enumerate(legal_moves):
        # Generate the description for each SubMove
        moves_description = ", ".join(
            f"[{'bar' if sub_move.start == -1 else sub_move.start}, "
            f"{'off' if sub_move.end == -2 else sub_move.end}, "
            f"{'*' if sub_move.hits_blot else '-'}]"
            for sub_move in move.sub_move_commands
        )

        # Retrieve the corresponding state value
        if state_values.dim() == 0:
            state_value = state_values.item()
        else:
            if i < state_values.size(0):
                state_value = state_values[i].item()
            else:
                state_value = state_values[-1].item()  # Handle any mismatch

        # Format the full move string with state value
        full_move_str = (
            f"Move {i}: {moves_description} | State Value: {state_value:.4f}"
        )

        # Print the move with its state value
        print(full_move_str)

    # Let the human select the agent's move
    selected_move = input(f"Select the agent's move (0 to {len(legal_moves) - 1}): ")
    if not selected_move.isdigit():
        print("Invalid input. Please enter a number.")
        return None

    selected_move = int(selected_move)
    if selected_move < 0 or selected_move >= len(legal_moves):
        print("Invalid move selected.")
        return None

    return selected_move


if __name__ == "__main__":
    print(sys.path)
    # To use the original automated agent selection, uncomment the line below:
    # play_game()

    # To use the human-selected agent moves, uncomment the line below:
    play_game_human_select_agent_moves()
