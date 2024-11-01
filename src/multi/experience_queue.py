class ExperienceQueue:
    def __init__(self, manager):
        self.queue = manager.Queue()

    def put(self, episode):
        self.queue.put(episode)

    def get(self, timeout=None):
        return self.queue.get(timeout=timeout)
