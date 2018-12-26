import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

action_size = 2
state_size = 4


class ActorCritic2(nn.Module) :
    def __init__(self) :
        super(ActorCritic2, self).__init__()

        self.dense_1 = nn.Linear(state_size, 128)
        self.probability = nn.Linear(128, action_size)
        self.value = nn.Linear(128, 1)

    def forward(self, x) :
        print("Data ", x, x.shape)
        out_1 = F.relu(self.dense_1(x))
        probs = self.probability(out_1)
        value = self.value(out_1)
        return F.softmax(probs, dim = -1), value


class Runner() :
    def __init__(self, env, gamma = 0.99, n_episode = 10000) :
        self.model = ActorCritic2()
        self.env = env
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-2)
        self.memory = []
        self.gamma = gamma
        self.log_delay = 50
        self.n_episode = n_episode
        self.batch_size = 32
        self.epsilon = np.finfo(np.float32).eps.item()

    def select_action(self, state) :
        state = torch.from_numpy(state).float()
        probs, value = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        self.memory.append((m.log_prob(action), value))
        return action.item()

    def replay(self, rewards_log) :
        memorized = self.memory
        policy_loss = []
        value_loss = []
        rewards = []
        R = 0
        for rew in rewards_log[::-1]: # Compute reward for trajectoire
            R = rew + self.gamma * R
            rewards.append(R)
        rewards.reverse()
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.epsilon) # Normalize
        for (log_prob, value), r in zip(memorized, rewards):
            Advantage = r - value.item() # Advantage estimate
            policy_loss.append(-log_prob * Advantage) # Policy gradient loss
            value_loss.append(F.smooth_l1_loss(value, torch.tensor([r]))) # Minimize R_T - prediction
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum() # Remove gradient, maintain loss value
        loss.backward()
        self.optimizer.step()
        self.memory = []



    def run(self) :
        rewards = []
        for i_episode in range(self.n_episode) :
            state = self.env.reset()
            ep_reward = 0
            for t in range(1000) : # Collect trajectoire
                action = self.select_action(state)
                state, reward, done, _ = env.step(action)
                ep_reward += reward
                rewards.append(reward)
                if done :
                    break
            self.replay(rewards)
            rewards = []
            if i_episode % self.log_delay == 0:
                print("Episode {} with reward : {}".format(i_episode, ep_reward))


if __name__ == '__main__' :
    env = gym.make('CartPole-v0')
    env.seed(42)
    torch.manual_seed(42)
    runner = Runner(env)
    runner.run()
