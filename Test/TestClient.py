from Environment import Client
import numpy as np
client = Client.Client(id = 5, n_item = 10, nb_color = 42, range_price =  120)
assert client.get_id == 5
assert len(client.get_taste_price) == 120
assert len(client.get_taste_color) == 42
assert np.sum(client.generate_proba_vision(10)) - 0.99 <= 0.02
assert len(client.generate_taste_price(120)) == 120
assert len(client.generate_taste_color(42)) == 42
