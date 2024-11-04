# experience_queue.py
from multiprocessing import Queue


class ExperienceQueue:
    def __init__(self):
        self.queue = Queue()

    def put(self, episode):
        self.queue.put(episode)

    def get(self, timeout=None):
        return self.queue.get(timeout=timeout)
