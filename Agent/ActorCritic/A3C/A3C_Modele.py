
import torch
import torch.nn as nn
import torch.nn.functional as F

from Agent.ActorCritic.A3C.utilities import set_init


class Net(nn.Module) :
    def __init__(self, s_dim, a_dim) :
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.dense1 = nn.Linear(s_dim, 64)
        self.out1 = nn.Linear(64, a_dim)
        self.dense2 = nn.Linear(s_dim, 64)
        self.out2 = nn.Linear(64, 1)
        set_init([self.dense1, self.out1, self.dense2, self.out2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x) :
        pi1 = F.relu(self.dense1(x))
        probabilities = self.out1(pi1)
        v1 = F.relu(self.dense2(x))
        values = self.out2(v1)
        return probabilities, values

    def choose_action(self, s) :
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim = 1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t) :
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim = 1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

