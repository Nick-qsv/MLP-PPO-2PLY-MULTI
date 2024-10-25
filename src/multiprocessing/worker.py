class Worker:
    def __init__(self, worker_id, parameter_manager):
        self.worker_id = worker_id
        self.parameter_manager = parameter_manager  # Shared parameters and version
        self.local_policy_network = None  # Local copy of policy network
        self.local_value_network = None  # Local copy of value network
        self.current_version = 0

    def run(self):
        while True:
            # Check for updated parameters
            if self.parameter_manager.version > self.current_version:
                self.update_local_parameters()
            # Run game and collect episodes
            episode = self.play_episode()
            # Upload episode to replay buffer
            # Continue loop
