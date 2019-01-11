"""
Reinforcement Learning (ActorCritic) using Pytroch + multiprocessing.
The most simple implementation for continuous action.
View more on my Chinese tutorial page [莫烦Python](https://morvanzhou.github.io/).
"""

import os

import gym
import torch.multiprocessing as mp

from Agent.ActorCritic.A3C.shared_adam import SharedAdam
from Agent.ActorCritic.A3C.Worker import Worker
from Agent.ActorCritic.A3C.A3C_Modele import Net
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
MAX_EP = 4000
os.environ["OMP_NUM_THREADS"] = "1"



env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


if __name__ == "__main__" :
    gnet = Net(N_S, N_A)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr = 0.0001)  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True :
        r = res_queue.get()
        if r is not None :
            res.append(r)
        else :
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()