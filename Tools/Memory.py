from collections import deque
import numpy as np

class Memory():

    def __init__(self, MaxSize = 10000):
        self.memory = deque(MaxSize)

    def append(self, elements):
        self.memory.append(elements)

    def sample(self, batch_size = 32):

        indices = np.random.choice(len(self.memory), range(batch_size), replace = False)
        states, actions, rewards, dones, next_states = zip(*[self.memory[idx] for idx in indices])
        return np.array(states),\
               np.array(actions),\
               np.array(rewards),\
               np.array(dones),\
               np.array(next_states)


