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
            self.list_client.append(Client(client_id))
        for item_id in range(n_item):
            self.list_items.append(Items(item_id))
            self.list_item_as_array.append(self.list_items[-1].get_as_list())
        self.current_client = -1



    def step(self, action):
        return 0

    def reset(self):
        self.current_client = np.random.randint(self.n_client)
        return self.list_client[self.current_client], self.list_item_as_array


