import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym

class Master(nn.Module):
    def __init__(self, gamma, max_ep, input_dim, action_dim):
        super(Master, self).__init__()
        self.gamma = gamma
        self.max_ep = max_ep
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.create_model()
        self.probability_distribution = torch.distributions.Categorical

    def create_model(self):
        self.dense1 = nn.Linear(self.input_dim, 30)
        self.dense2 = nn.Linear(30, self.action_dim)

    def forward(self, x):
        out_1 = torch.nn.functional.relu(self.dense1(x))
        output = torch.nn.functional.linear(out_1)




