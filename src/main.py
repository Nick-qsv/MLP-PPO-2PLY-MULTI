import multiprocessing
from multi import ParameterManager, Worker, ExperienceQueue, worker_function
from agents import Trainer, BackgammonPolicyNetwork
from utils import RingReplayBuffer
from config import *
import time
import torch
import queue
import pynvml


def main():
    """
    Main function to initialize and execute the distributed training pipeline.

    This function performs the following operations:

    1. **Initialization**:
        - Sets up the `ParameterManager` to manage shared model parameters and versioning.
        - Initializes a `RingReplayBuffer` with a maximum size of 10,000 to store episodes from workers.
        - Creates an `ExperienceQueue` to collect experiences from worker processes.
        - Initializes a `Trainer` instance responsible for updating the policy and value networks.

    2. **Worker Processes**:
        - Spawns 7 worker processes, each running an instance of the PPO agent.
        - Each worker interacts with the environment to generate game episodes.
        - Workers upload experiences grouped into episodes to the `ExperienceQueue`.
        - Workers periodically poll the `ParameterManager` for updated parameters based on the version number and update their local copies if a new version is available.

    3. **Training Loop**:
        - Continuously monitors the `ExperienceQueue` to retrieve and store episodes in the `RingReplayBuffer`.
        - Once the `RingReplayBuffer` accumulates 1,000 episodes, it serializes the data and triggers the `Trainer` to perform a network update on the GPU.
        - After training, the `Trainer` updates the shared `ParameterManager` with the new `state_dict` and increments the version number to notify workers of the update.
        - Every 100,000 episodes, the current model is saved to S3 for backup and checkpointing purposes.
        - The training loop continues until a total of 1 million episodes have been processed.

    4. **Concurrency and Synchronization**:
        - Utilizes a `multiprocessing.Manager` to maintain a thread-safe `state_dict` of model parameters and a shared version number.
        - The `Trainer` acquires a lock on the `ParameterManager` only during parameter updates to ensure thread safety.
        - Workers read from the `ParameterManager` without locking, as they only update their local copies of the parameters.

    5. **Termination**:
        - After reaching the target number of episodes, the function gracefully terminates all worker processes.

    **Components**:
        - `ParameterManager`: Manages shared model parameters and versioning across processes.
        - `RingReplayBuffer`: Stores a fixed number of episodes for training.
        - `ExperienceQueue`: Queue for collecting experiences from workers.
        - `Trainer`: Handles the training of policy and value networks using collected experiences.
        - `Worker`: Represents individual worker processes running PPO agents.

    **Device Configuration**:
        - The `Trainer` is configured to use a GPU if available (`cuda`), otherwise defaults to CPU.
        - The policy network is moved to the configured device for efficient computation.

    **Model Saving**:
        - Models are periodically saved to S3 to ensure progress is not lost and to allow for checkpointing.

    **Termination Condition**:
        - The training process stops once 1 million episodes have been processed.

    """
    pynvml.nvmlInit()
    # Initialize multiprocessing manager
    manager = multiprocessing.Manager()

    # Create shared objects using the manager
    lock = manager.Lock()
    version = manager.Value("i", 1)
    parameters = manager.dict()

    # Initialize the policy network and get its state_dict
    initial_network = BackgammonPolicyNetwork()
    state_dict = initial_network.state_dict()

    # Convert tensors to NumPy arrays for serialization
    for key, tensor in state_dict.items():
        parameters[key] = tensor.cpu().numpy()

    # Initialize parameter manager with shared objects
    parameter_manager = ParameterManager(lock, version, parameters)

    # Initialize ring replay buffer and experience queue
    replay_buffer = RingReplayBuffer(max_size=10000)
    experience_queue = ExperienceQueue()

    # Create and start worker processes
    worker_processes = []
    for i in range(7):
        worker_process = multiprocessing.Process(
            target=worker_function, args=(i, parameter_manager, experience_queue)
        )
        worker_process.start()
        worker_processes.append(worker_process)

    # Device setup for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Initialize Trainer (no separate process)
    trainer = Trainer(
        parameter_manager=parameter_manager,
        device=device,
        s3_bucket_name=S3_BUCKET_NAME,
        s3_log_prefix=S3_LOG_PREFIX,
    )

    # Initialize trainer's state
    state_dict = parameter_manager.get_parameters()
    trainer.policy_network.load_state_dict(state_dict)

    # Monitor training and handle model saving
    episode_count = 0

    # Initialize episode time tracker
    last_print_time = time.time()

    while episode_count < NUM_EPISODES:
        # Try to get episodes from the ExperienceQueue
        try:
            episodes = experience_queue.get(
                timeout=1
            )  # Now expecting a list of episodes
            for episode in episodes:
                replay_buffer.add_episode(episode)
                episode_count += 1
                if episode_count % 100 == 0 and episode_count != 0:
                    print(
                        f"Episode count incremented to: {episode_count}"
                    )  # Debug statement

            # Check if replay_buffer has reached x episodes
            if len(replay_buffer.buffer) >= MIN_EPISODES_TO_TRAIN:
                # Drain the buffer and push episodes to the Trainer
                episodes_to_train = list(replay_buffer.buffer)
                replay_buffer.buffer.clear()
                # Convert episodes to tensors on the GPU
                for episode in episodes_to_train:
                    episode.to_tensor(device=device)

                # Perform training
                trainer.update(episodes_to_train)

        except queue.Empty:
            # No new episodes in the ExperienceQueue
            print("Experience queue is empty. Waiting for episodes...")
            pass

        if episode_count % 300 == 0 and episode_count != 0:
            current_time = time.time()
            elapsed = current_time - last_print_time
            eps_per_sec = 200 / elapsed if elapsed > 0 else float("inf")
            print(
                f"***** Episode {episode_count} completed | {eps_per_sec:.2f} eps/sec ****"
            )
            last_print_time = current_time

        # # Test save to make sure S3 working
        # if episode_count == 100:
        #     print(f"Saving model at episode {episode_count}")
        #     filename = f"backgammon_test_episode_{episode_count}.pth"
        #     # Save model via ParameterManager
        #     parameter_manager.save_model(filename=filename, to_s3=True)

        # Check if it's time to save the model
        if episode_count % MODEL_SAVE_FREQUENCY == 0 and episode_count != 0:
            filename = f"backgammon_128_guided_episode_{episode_count}.pth"
            # Save model via ParameterManager
            parameter_manager.save_model(filename=filename, to_s3=True)

    # Terminate worker processes after training
    for worker_process in worker_processes:
        worker_process.terminate()

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Start method has already been set
        pass
    main()
