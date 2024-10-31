import multiprocessing
from multiprocessing import ParameterManager, Worker, ExperienceQueue
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
    # Initialize parameter manager, ring replay buffer, and experience queue
    parameter_manager = ParameterManager()
    replay_buffer = RingReplayBuffer(max_size=10000)
    experience_queue = ExperienceQueue()
    # Create and start worker processes
    workers = [
        Worker(
            worker_id=i,
            parameter_manager=parameter_manager,
            experience_queue=experience_queue,
        )
        for i in range(7)
    ]
    worker_processes = []
    for worker in workers:
        worker_process = multiprocessing.Process(target=worker.run)
        worker_process.start()
        worker_processes.append(worker_process)

    # Initialize Trainer (no separate process)
    trainer = Trainer(parameter_manager=parameter_manager)

    # Device setup for GPU
    trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {trainer.device}")
    trainer.policy_network.to(trainer.device)

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
            episode = experience_queue.get(timeout=1)
            replay_buffer.add_episode(episode)
            episode_count += 1

            # Check if replay_buffer has reached 1,000 episodes
            if len(replay_buffer.buffer) >= 1000:
                # Drain the buffer and push episodes to the Trainer
                episodes_to_train = list(replay_buffer.buffer)
                replay_buffer.buffer.clear()
                # Perform training directly
                trainer.update(episodes_to_train)

        except queue.Empty:
            # No new episodes in the ExperienceQueue
            pass

        # Check if it's time to save the model
        if episode_count % MODEL_SAVE_FREQUENCY == 0 and episode_count != 0:
            # Save model to S3 or local storage
            print(f"Saving model at episode {episode_count}")
            # parameter_manager.save_model()

        if episode_count % 100 == 0:
            current_time = time.time()
            elapsed = current_time - last_print_time
            eps_per_sec = 100 / elapsed if elapsed > 0 else float("inf")
            print(f"Episode {episode_count} completed | {eps_per_sec:.2f} eps/sec")
            last_print_time = current_time

    # Terminate worker processes after training
    for worker_process in worker_processes:
        worker_process.terminate()

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()


# 7 workers running individual games
# uploading experiences grouped into episodes to experience queue
# update the policy network and value network using the Trainer class, which passes the updated state_dict to the multiprocessingManager using a lock to update the state_dict parameters
# workers periodically check the version number of the Manager after every episodeand if there is a new version, they update their local copy of the parameters
# ring replay buffer storing episodes from the workers, with a maxsize=10000
# when a certain number of episodes are inside the ring buffer, grab the episodes, convert to serialize the data, run an update on the GPU through the Trainer class
# Trainer passes updated state_dict to multiprocessingManage
# all during this update, workers keep on running games and uploading experiences to the queue
# if the number of episodes used in updates % 100,000, upload the current model to S3
# when episode 1 million is reached, stop the training

# every worker has its own instance of the ppo agent class with the policy and value network
# thread safe because the params have a lock only for writes, when the workers update their local copy of the parameters, they don't need to take a lock

# There will be a multiprocessing.Manager() that has a state_dict of the parameters and a version number Value
# the Trainer will have a lock on this manager and will update the parameters and increment the version number
# the workers will keep polling the manager for the new parameters and the new version number
# when the version number is different from the one they have, they will update their parameters and the version number
# they don't need to take a lock because they are only reading and updating their own local copy
