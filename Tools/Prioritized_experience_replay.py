import numpy as np
import torch
 # https://github.com/qfettes/DeepRL-Tutorials
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PrioritizedReplayMemory() :
    def __init__(self, maxlen = 1000, alpha = 0.6, beta = 0.4, beta_annealing = 1000):
        self.maxlen = maxlen
        self.buffer = []
        self.pos = 0
        self.iteration = 0
        self.alpha = alpha # Randomness if 0, 1 pure highest priority
        self.beta = beta # Importance sampling
        self.beta_start = beta
        self.beta_annealing = beta_annealing
        self.priorities = np.zeros((maxlen,), dtype = np.float32)

    def remember(self, data):
        max_prio = np.max(self.priorities) if len(self.buffer) > 0 else 1.0 ** self.alpha
        if len(self.buffer) < self.maxlen:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.maxlen

    def sample(self, batch_size):
        if len(self.buffer) == self.maxlen:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probas = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p = probas)
        samples = [self.buffer[idx] for idx in indices]
        self.beta = self.update_beta(self.iteration)
        self.iteration += 1
        proba_min = np.min(probas)
        max_weight = (proba_min * probas[indices]) ** (-self.beta)
        weights = (len(self.buffer) * probas[indices]) ** (-self.beta)
        weights /= max_weight # Normalisation - Divise par max donc de 0 a 1, si divise par sum alors la somme de tous est 1
        weights = torch.tensor(weights, device = device, dtype = torch.float)
        return samples, indices, weights

    def update_beta(self, iteration):
        return min(1.0, self.beta_start + iteration * (1.0 - self.beta_start) / self.beta)

    def update_priorities(self, batch_indices, batch_prio):
        for idx, priority in zip(batch_indices, batch_prio):
            self.priorities[idx] = (priority + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.buffer)
