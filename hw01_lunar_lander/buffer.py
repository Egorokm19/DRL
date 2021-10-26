from random import sample
from collections import deque

        
class Buffer:
    def __init__(self):
        self.buffer = deque(maxlen=10000)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return sample(self.buffer, batch_size)
    