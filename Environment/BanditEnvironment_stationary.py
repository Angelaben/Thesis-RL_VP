import numpy as np
from Environment.Client import Client
from Environment.Items import Items
import random
# Cas stationaire : Les items ne changent pas
class BanditEnvironment:
    def __init__(self, n_client = 10, n_item = 10, nb_color = 5, range_price = 5):
        self.list_items = []
        self.list_client = []
        self.list_item_as_array = []
        self.n_client = n_client
        self.n_item = n_item
        for client_id in range(n_client) :
            self.list_client.append(Client(client_id, n_item = n_item, nb_color = nb_color, range_price = range_price))
        for item_id in range(n_item):
            self.list_items.append(Items(item_id, interval_price = range_price, interval_color = nb_color))
            self.list_item_as_array.append(self.list_items[-1].get_as_list())
        self.current_client = -1
        self.range_price = range_price
        self.nb_color = nb_color


# Recupere item le plus apprecie et faire histogramme des recos ?
# Imposer un client ?

    # Action is a simple item
    def step_mono_recommendation(self, action):
        item_selected = self.list_items[action]
      #  print("Recommande ", action, " a l'utilisateur ", self.current_client)
       # print("Client courant : centre prix - taste", self.list_client[self.current_client].get_properties)
        reward = self.list_client[self.current_client].offer(item_selected)
        self.current_client = np.random.randint(self.n_client)
        # Non content based
        random.shuffle(self.list_items)
        return self.current_client, self.list_items , reward

    # Action is a list of item
    def step_list_recommendation(self, action):
        return 0

    def reset(self):
        self.current_client = np.random.randint(self.n_client)
        # Non content based
        return self.current_client, self.list_items


