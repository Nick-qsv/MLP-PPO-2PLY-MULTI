import multiprocessing
from multiprocessing import ParameterManager, Worker, ExperienceQueue
from agents import Trainer, BackgammonPolicyNetwork
from utils import RingReplayBuffer
from config import *


def main():
    # Initialize parameter manager, ring replay buffer, and experience queue
    parameter_manager = ParameterManager()
    replay_buffer = RingReplayBuffer(max_size=10000)
    experience_queue = ExperienceQueue()
    trainer_queue = multiprocessing.Queue()

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

    # Create and start trainer process
    trainer = Trainer(parameter_manager=parameter_manager, trainer_queue=trainer_queue)
    trainer_process = multiprocessing.Process(target=trainer.train)
    trainer_process.start()

    # Monitor training and handle model saving
    episode_count = 0

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
                # Send episodes to Trainer via trainer_queue
                trainer_queue.put(episodes_to_train)
        except queue.Empty:
            # No new episodes in the ExperienceQueue
            pass

        # Check if it's time to save the model
        if episode_count % MODEL_SAVE_FREQUENCY == 0 and episode_count != 0:
            # Save model to S3 or local storage
            print(f"Saving model at episode {episode_count}")
            # parameter_manager.save_model()

    # Terminate processes after training
    for worker_process in worker_processes:
        worker_process.terminate()
    trainer_process.terminate()


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
