import numpy as np

np.random.seed(84)


class Client :
    # Feature : Price - Color - Gender
    def __init__(self, id, n_item,  nb_color = 5, range_price = 100) :
        self._my_id = id
        self.center_of_price = np.random.randint(
            range_price)  # Le prix parfait pour le client est une loi normle centre sur ce random
        # De cette facon le client n'accepte qu'une gamme de prix sinon il "refuse" que celle ci soit trop cheap / chere

        self._taste = [
            # Price
            self.generate_taste_price(range_price),
            self.generate_taste_color(nb_color),
            np.random.randint(2)
        ]
        self.proba_vision = self.generate_proba_vision(n_item) # Pas encore utilisé

    def generate_proba_vision(self, list_size) :
        rand = np.random.random(list_size) * 100
        rand = rand / np.sum(rand)
        rand.sort()
        rand = rand.tolist()
        rand.reverse()
        return rand

    def generate_taste_price(self, range_price) :
        priced = np.array(range(range_price))
        sigma = np.std(priced)
        prices = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(
            -(priced - self.center_of_price) ** 2 / (2 * sigma ** 2) * 2)  # Suit loi normale
        return prices

    def generate_taste_color(self, nb_color) :
        colors = np.random.random(nb_color) * 100
        colors /= np.sum(colors)
        return colors

    @property
    def get_properties(self) :
        return self.center_of_price, self._taste

    @property
    def get_id(self) :
        return self._my_id

    @property
    def get_taste_price(self) :
        return self._taste[0]

    @property
    def get_taste_color(self) :
        return self._taste[1]

    @property
    def get_proba_vision(self) :
        return self.proba_vision

    def offer(self, item_recommended, is_list_reco = False) :
        reward = 0
        coin_toss = np.random.random()
        if is_list_reco :
            return -1  # TODO
        else :
            price, color, gender = item_recommended.get_properties
            proba_buy_price = self._taste[0][price]
            proba_buy_color = self._taste[1][color]
            proba_buy_gender = self._taste[2]  # Ignore tant que je trouve une facon de l'utiliser
            return price if coin_toss < proba_buy_color * proba_buy_price else 0
        # Ou faire P(A) + P(b) - P(A int B) ?
