from collections import namedtuple, deque
import random


Transition = namedtuple('Transition', ('state', 'o_action', 'd_action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity, batch_size):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)
