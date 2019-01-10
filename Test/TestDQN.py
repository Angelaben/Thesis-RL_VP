from Agent.RecommenderMono_Stationary_DQN_NonContentBased import Runner, DQN, preprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from IPython.display import clear_output
from Environment.BanditEnvironment_stationary import BanditEnvironment as env_generator
from Tools import Logger
from keras.utils import to_categorical
from Environment import Client
from Environment import Items


n_feature_item = 3
n_feature_client = 3
range_price = 10
range_color = 5
n_client = 5
n_item = 5
input_item_size = 10
input_client_size = 10
hidden_layer_size = 256
hidden_layer_2_size = 256
learning_rate = 1e-5
log_delay = 100
batch_size = 256
epsilon_decay = 1000
item = Items.Items(4, range_price - 1, range_color - 1, catalog_size = 10)
client = Client.Client(id = 4, n_item = n_item, nb_color = range_color, range_price =  range_price)

env = env_generator(n_client = n_client, n_item = n_item, nb_color = range_color, range_price = range_price)
runner = Runner(env)
state = preprocess(user_id = client.get_id, items = [item])
print(state)
for i in range(100):
    client_id, items = env.reset()
    state = [client_id, items]
    action = runner.model.choose_action(state, 0.01)
    print(action)