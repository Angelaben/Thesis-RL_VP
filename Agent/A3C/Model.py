import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
from itertools import count
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from tqdm import trange
from collections import deque

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.create_model()
        self.distribution = torch.distributions.Categorical

    def create_model(self):
        self.dense_1 = torch.nn.Linear(self.state_dim, 128)
        self.policy = torch.nn.Linear(128, self.action_dim)

        self.dense_2 = torch.nn.Linear(self.state_dim, 128)
        self.values = torch.nn.Linear(128, 1)

    def forward(self, data):
        out_1 = self.dense_1(data)
        act_1 = torch.nn.functional.relu(out_1)
        policy = torch.nn.functional.relu(self.policy(act_1))

        out_2 = self.dense_2(data)
        act_2 = torch.nn.functional.relu(out_2)
        values = torch.nn.functional.relu(self.values(act_2))

        return policy, values
