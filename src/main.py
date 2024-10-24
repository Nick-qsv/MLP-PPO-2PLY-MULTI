# 7 workers running individual games
# uploading experiences grouped into episodes to experience queue
# after 100 games, they check if there was an update to the policy and value network by polling the update queue
# ring replay buffer storing episodes and maxsize=10000
# when a certain number of episodes are inside the ring buffer
# grab the episodes, convert to appropriate data, run an update on the GPU
# update the policy network and value network
# communicate this update to the workers through the update queue
# all during this update, workers keep on running games and uploading experiences to the queue
# if the number of episodes used in updates % 100,000, upload the current model to S3
# when episode 1 million is reached, stop the training

# every worker has its own instance of the ppo agent class with the policy and value network
# this agent class has the parameters stored in some other class that gets atomic swap updated by the Trainer
# thread safe because the params have a lock only for writes not readss from workers to the parameters
