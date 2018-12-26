import os

import gym
import torch.multiprocessing as mp

from Agent.A3C.utilities import convert_tensor, train_agent, record
from Agent.A3C.A3C_Modele import Net
env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
MAX_EP = 4000
os.environ["OMP_NUM_THREADS"] = "1"


class Worker(mp.Process) :
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name) :
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.globalNet, self.opt = gnet, opt
        self.localNet = Net(N_S, N_A)  # local network
        self.env = gym.make('CartPole-v0').unwrapped

    def run(self) :
        total_step = 1
        while self.g_ep.value < MAX_EP :
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True :
                a = self.localNet.choose_action(convert_tensor(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done :
                    r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done :  # update global and assign to local net
                    # sync
                    train_agent(self.opt, self.localNet, self.globalNet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done :  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)
