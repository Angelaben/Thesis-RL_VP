import numpy as np

from Environment.BanditEnvironment_stationary import BanditEnvironment
from Environment.Client import Client
from Environment.Items import Items


# Cas stationnaire : Tout est regenere, pas de modif au cours du temps apart regeneration de nouveaux items et client
#
class BanditEnvironment_regeneration(BanditEnvironment) :
    def __init__(self, n_client = 10, \
                 n_item = 10, \
                 nb_color = 5, \
                 range_price = 5, \
                 catalog_size = 50, \
                 regeneration_delay = 1000) :
        super(BanditEnvironment_regeneration, self).__init__(n_client, n_item, nb_color, range_price, catalog_size)
        self.regeneration_delay = regeneration_delay
        self.count = 0

    def step_mono_recommendation(self, action) :

        self.current_client, self.list_items, reward = super(BanditEnvironment_regeneration, self).step_mono_recommendation(action)
        self.count += 1  # La regeneration de client et d'item n'a pas lieu dans la reco
        return self.current_client, self.list_items, reward
    def reset(self) :
        self.current_client, self.list_items = super(BanditEnvironment_regeneration, self).reset()
        self.count += 1
        should_reset = False
        if self.count > self.regeneration_delay :
            self.count = 0
            self.regenerate()
            should_reset = True
        return self.current_client, self.list_items, should_reset  # Should reset pour le logger

    def regenerate(self) :
        print("Regeneration of clients and items")
        self.list_items = []
        self.catalog = []
        self.list_client = []
        for client_id in range(self.n_client) :
            self.list_client.append(
                Client(client_id, n_item = self.n_item, nb_color = self.nb_color, range_price = self.range_price))
        for item_id in range(self.catalog_size) :
            self.catalog.append(Items(item_id, interval_price = self.range_price, interval_color = self.nb_color))
        indices = np.random.randint(0, self.catalog_size, self.n_item)  # Genere une nouvelle liste
        self.list_items = [self.catalog[idx] for idx in indices]
