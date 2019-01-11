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

BATCH_SIZE = 32
class DQN(nn.Module) :
    def __init__(self, gamma, max_ep, input_dim, action_dim) :
        super(DQN, self).__init__()
        self.gamma = gamma
        self.max_ep = max_ep
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.create_model()



    def create_model(self) :
        self.dense1 = nn.Linear(self.input_dim, 30)
        self.dense2 = nn.Linear(30, self.action_dim)

    def forward(self, x) :
        x = torch.autograd.Variable(torch.from_numpy(x)).float()
    #    print("Forward ", x, x.shape)
        out_1 = torch.relu(self.dense1(x))
        return self.dense2(out_1)

def remember(memory, state, action, next_state, reward, done):
    memory.append((state, action, next_state, reward, done))

def sample(batch_size, memory):
    indices = np.random.choice(len(memory), batch_size, replace = False)
    states, actions, next_states, rewards, dones = zip(*[memory[idx] for idx in indices])
    return np.array(states), torch.Tensor(actions), np.array(next_states), np.array(rewards), np.array(dones)






def choose_action(state, epsilon, model):

    coin_toss = np.random.random()
    if coin_toss < epsilon :
        return np.random.randint(2)
    else :
        return torch.argmax(model(state)[0]).item()


def replay(memory, modele, optimizer):
    states, actions, next_states, rewards, dones = sample(BATCH_SIZE, memory)
    prediction = torch.Tensor((1 - dones) * (rewards + modele.gamma * np.amax(target(next_states).detach().max(1)[0].numpy(), axis = 1)))
    prediction = prediction.view(-1, 1)
    real_value = modele(states)[:, 0, :]
    actions = actions.view(-1, 1)
    print("Actions ", actions.shape)
    real_value = torch.Tensor(real_value).gather(1, actions.long())

    loss = F.mse_loss(real_value, prediction.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

memory = deque(maxlen = 10000)
model = DQN(
    gamma = 0.99,
    max_ep = 400,
    input_dim = 4,
    action_dim = 2
)
print(model)

target = DQN(
    gamma = 0.99,
    max_ep = 400,
    input_dim = 4,
    action_dim = 2
)
target.load_state_dict(model.state_dict())
target.eval()
optimizer = optim.Adam(model.parameters(), lr = 5e-2)
env = gym.make('CartPole-v0')#.unwrapped
epsilon = 1.0
epsilon_origin = 1.0
decay_period = 200
plotter = []
mean_plot = []
for i in trange(1, 400):
    state = env.reset()
    state = np.reshape(state, (1, 4))
    cumul_reward = 0
    done = False
    counter = 0
    while not done:
        action = choose_action(state, epsilon, model)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, (1, 4))
        cumul_reward += reward
        counter += 1
        remember(memory = memory, state = state, action = action, reward = reward, next_state = next_state, done = done)
        if len(memory) > BATCH_SIZE:
            replay(memory = memory, modele = model, optimizer = optimizer)
        state = next_state
        epsilon = max(0.05, epsilon_origin - i / decay_period * epsilon_origin)
    plotter.append(cumul_reward)
    mean_plot.append(np.mean(plotter))
    if (i + 1) % 10 == 0 :
        target.load_state_dict(model.state_dict())
    if i % 50 == 0:
        print("Reward %d at episode %d with epsilon %f " % (cumul_reward, i, epsilon))

plt.plot(plotter)
plt.plot(mean_plot)
plt.grid()
plt.show()


