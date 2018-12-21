import torch.multiprocessing as mp
from Agent.A3C import Model
import gym


class Worker(mp.Process) :
    global_episode_counter = 0

    def __init__(self, masterModel, optimizer, global_ep, global_ep_r, res_queue, name, state_dim, action_dim) :
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.masterModel = masterModel,
        self.optimizer = optimizer
        self.local_model = Model.ActorCritic(state_dim, action_dim)  # local network
        self.env = gym.make('CartPole-v0').unwrapped

    def run(self) :
    