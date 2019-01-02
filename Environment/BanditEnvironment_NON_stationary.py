import random

import numpy as np

# Cas stationaire : Les items ne changent pas
from Environment.BanditEnvironment_stationary import BanditEnvironment
from Environment.Client import Client
from Environment.Items import Items


# Regenere une liste d'item toutes les N iterations
class BanditEnvironment_Non_stationary(BanditEnvironment) :
    def __init__(self, n_client = 10, n_item = 10, nb_color = 5, range_price = 5, \
                 interval_regeneration = 1000, NS_client = True, NS_items = False) :
        super(BanditEnvironment_Non_stationary, self).__init__(n_client, n_item, nb_color, range_price)
        self.regeneration = interval_regeneration
        self.NS_Client = NS_client
        self.NS_items = NS_items
        self.iterator_change = 0

    # Action is a simple item
    def step_mono_recommendation(self, action) :
        self.iterator_change += 1
        item_selected = self.list_items[action]
        reward = self.list_client[self.current_client].offer(item_selected)
        self.current_client = np.random.randint(self.n_client)
        # Non content based
        random.shuffle(self.list_items)
        if self.NS_Client and self.iterator_change % self.regeneration == 0 :
            self.generate_new_list_of_clients()
        if self.NS_items and self.iterator_change % self.regeneration == 0 :
            self.generate_new_list_of_item()

        return self.current_client, self.list_items, reward

    # Action is a list of item
    def step_list_recommendation(self, action) :
        return 0

    def reset(self) :
        self.current_client = np.random.randint(self.n_client)
        self.iterator_change = 0
        self.generate_new_list_of_item()
        self.generate_new_list_of_clients()
        return self.current_client, self.list_items

    def generate_new_list_of_item(self) :
        self.list_items = []
        for cpt in range(self.n_item) :
            self.list_items.append(Items(cpt, interval_price = self.range_price, interval_color = self.nb_color))
            self.list_item_as_array.append(self.list_items[-1].get_as_list())

    def generate_new_list_of_clients(self) :
        self.list_client = []
        for client_id in range(self.n_client) :
            self.list_client.append(Client(client_id,n_item = self.n_item,  nb_color = self.nb_color, range_price = self.range_price))


