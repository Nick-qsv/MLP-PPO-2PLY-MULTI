import multiprocessing


class ExperienceQueue:
    def __init__(self):
        self.queue = multiprocessing.Queue()

    def put(self, episode):
        self.queue.put(episode)

    def get(self):
        return self.queue.get()
