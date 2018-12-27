import numpy as np
from Environment.Client import Client
from Environment.Items import Items

class BanditEnvironment:
    def __init__(self, n_client = 10, n_item = 10):
        self.list_items = []
        self.list_client = []
        self.list_item_as_array = []
        self.n_client = n_client
        self.n_item = n_item
        for client_id in range(n_client) :
            self.list_client.append(Client(client_id, range_price = 5))
        for item_id in range(n_item):
            self.list_items.append(Items(item_id, interval_price = 5))
            self.list_item_as_array.append(self.list_items[-1].get_as_list())
        self.current_client = -1



    # Action is a simple item
    def step_mono_recommendation(self, action):
        item_selected = self.list_items[action]
        print("Recommande ", action, " a l'utilisateur ", self.current_client)
        print("Client courant : centre prix - taste", self.list_client[self.current_client].get_properties)
        self.list_client[self.current_client].offer(item_selected)
        self.current_client = np.random.randint(self.n_client)

        return 0

    # Action is a list of item
    def step_list_recommendation(self, action):
        return 0

    def reset(self):
        self.current_client = np.random.randint(self.n_client)
        return self.list_client[self.current_client], self.list_item_as_array


