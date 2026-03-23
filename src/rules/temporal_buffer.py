from collections import deque


class TemporalBuffer:
    def __init__(self, maxlen=30):
        self.history = deque(maxlen=maxlen)

    def update(self, item):
        self.history.append(item)

    def get_recent(self):
        return list(self.history)