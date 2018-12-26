import numpy as np
np.random.seed(84)

class Client:
    # Feature : Price - Color - Gender
    def __init__(self, id, nb_color = 5, range_price = 100):
        self._my_id = id
        self.center_of_price = np.random.randint(range_price) # Le prix parfait pour le client est une loi normle centre sur ce random
        # De cette facon le client n'accepte qu'une gamme de prix sinon il "refuse" que celle ci soit trop cheap / chere

        self._taste = [
            # Price
            self.generate_taste_price(range_price),
            self.generate_taste_color(nb_color),
            np.random.randint(2)
        ]


    def generate_taste_price(self, range_price):
        priced = np.array(range(range_price))
        sigma = np.std(priced)
        prices = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(priced - self.center_of_price) ** 2 / (2 * sigma ** 2) * 2) # Suit loi normale
        return prices

    def generate_taste_color(self, nb_color):
        colors = np.random.random(nb_color) * 100
        colors /= np.sum(colors)
        return colors