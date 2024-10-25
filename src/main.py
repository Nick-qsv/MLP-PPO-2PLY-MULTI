import multiprocessing
from multiprocessing import ParameterManager, Worker, ExperienceQueue
from agents import Trainer, BackgammonPolicyNetwork
from utils import RingReplayBuffer
from config import *


def main():
    # Initialize multiprocessing manager
    parameter_manager = ParameterManager()

    # Initialize ring replay buffer
    replay_buffer = RingReplayBuffer(max_size=10000)

    # Create and start worker processes
    workers = [
        Worker(worker_id=i, parameter_manager=parameter_manager) for i in range(7)
    ]
    for worker in workers:
        worker_process = multiprocessing.Process(target=worker.run)
        worker_process.start()

    # Create and start trainer process
    trainer = Trainer(parameter_manager=parameter_manager, replay_buffer=replay_buffer)
    trainer_process = multiprocessing.Process(target=trainer.train)
    trainer_process.start()

    # Monitor training and handle model saving
    episode_count = 0
    while episode_count < NUM_EPISODES:
        # Check if it's time to save the model
        if episode_count % MODEL_SAVE_FREQUENCY == 0:
            # Save model to S3
            pass
        # Update episode_count based on episodes processed
        pass

    # Terminate processes after training
    for worker in workers:
        worker_process.terminate()
    trainer_process.terminate()


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
