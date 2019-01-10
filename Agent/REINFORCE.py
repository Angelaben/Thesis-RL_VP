
import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# REINFORCE ALGORITHM :
# Policy gradient
# For episode (s, a, st+1, r)
    # for t in time :
        #Weight <= Weight + alpha Grad(log(pi_(s_t,a_t))v_t
gamma = 0.99
np.random.seed(42)
torch.manual_seed(42)
log_interval = 10
env = gym.make('CartPole-v0')
env.seed(42)
hidden_layer_size = 128
state_size = 4
action_size = 2
learning_rate = 1e-2

class Policy(nn.Module) :
    def __init__(self) :
        super(Policy, self).__init__()
        self.dense1 = nn.Linear(state_size, hidden_layer_size)
        self.dense2 = nn.Linear(hidden_layer_size, action_size)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x) :
        x = F.relu(self.dense1(x))
        action_scores = self.dense2(x)
        return F.softmax(action_scores, dim = 1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr = learning_rate)
eps = np.finfo(np.float32).eps.item() # Mini delta
print("Environment threshold ", env.spec.reward_threshold)

def select_action(state) :
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode() :
    cumulative_decayed_reward = 0
    policy_loss = []
    rewards = []
    for reward in policy.rewards[::-1] :
        cumulative_decayed_reward = reward + gamma * cumulative_decayed_reward
        rewards.insert(0, cumulative_decayed_reward)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps) # Normalisation
    for log_prob, reward in zip(policy.saved_log_probs, rewards) :
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main() :
    running_reward = 10
    for i_episode in count(1) :
        state = env.reset()
        for t in range(10000) :  # collect trajectoire
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            if done :
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % log_interval == 0 :
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold :
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__' :
    main()
