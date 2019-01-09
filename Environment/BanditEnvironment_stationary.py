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
        reward = self.list_client[self.current_client].offer(item_selected)
        self.current_client = np.random.randint(self.n_client)
        # Non content based
      #  random.shuffle(self.list_items)
        return self.current_client, self.list_items , reward

    # Action is a list of item
    def step_list_recommendation(self, action):
        return 0

    def reset(self):
        self.current_client = np.random.randint(self.n_client)
        # Non content based
        return self.current_client, self.list_items


    # Recuperer l'esperance de gain en recuperant le max esperance max item par client , moyenne et non moyenne sur les autres items
    # Reward / Client = Max proba item * prix (= 1 pour l'instant, juste proba d'acaht sans reward sur l'achat)
    def get_indicator(self) :
        max_reward_probs = []
        for client in self.list_client :
            # Calcul appetence de l'user pour la liste d'item
            taste_price = client.get_taste_price
            taste_color = client.get_taste_color
            max_item_score = -1
            for counter, item in enumerate(self.list_items) :
                score_price = taste_price[item.get_Price]  # Esperance achat selon prix
                score_color = taste_color[item.get_color]  # Esperance achat selon couleur
                expected_reward_proba = score_price * score_color * item.get_Price
                if expected_reward_proba > max_item_score :
                    max_item_score = expected_reward_proba
            max_reward_probs.append(max_item_score)  # Item le plus probable d'achat
        return max_reward_probs
