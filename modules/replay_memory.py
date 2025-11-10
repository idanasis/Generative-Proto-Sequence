

# Replay Memory:
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('observations', 'next_observations', 'actions', 'rewards', 'dones',
                                       'actor_proto_plan_emb'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=self.capacity)
        self.hash_d = {}

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def empty(self):
        self.memory = deque([], maxlen=self.capacity)
        self.hash_d = {}

    def __len__(self):
        return len(self.memory)