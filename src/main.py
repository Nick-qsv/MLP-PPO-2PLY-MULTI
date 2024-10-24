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
