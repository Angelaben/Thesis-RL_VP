import numpy as np

np.random.seed(84)

# Rajouter decaying proba item
# Reco list
class Client :
    # Feature : Price - Color - Gender
    def __init__(self, id, n_item, nb_color = 5, range_price = 100) :
        self._my_id = id
        self.n_item = n_item

        self.center_of_price = np.random.randint(
            range_price)  # Le prix parfait pour le client est une loi normle centre sur ce random
        # De cette facon le client n'accepte qu'une gamme de prix sinon il "refuse" que celle ci soit trop cheap / chere

        self._taste = [
            # Price
            self.generate_normal_distirbution(0, range_price, self.center_of_price),
            self.generate_taste_color(nb_color),

        ]  # A priori sur les items

        # Caracteristics
        self.is_parent = np.random.random() < 0.1
        self.age = np.random.randint(18, 100)
        self.is_married = np.random.random() < 0.5
        self.gender = np.random.random() < 0.5  # True = Femme, False = homme
        self.caracteristics = [
            self.is_parent,
            self.age,
            self.is_married,
            self.gender
        ]

        # Proba caracteristics
        coin_toss = np.random.random()
        if self.is_parent :
            self.taste_is_parent = [min(coin_toss, 1 - coin_toss),
                                    max(coin_toss, 1 - coin_toss)]  # Proba regard objet pas parent / parent
        else :
            self.taste_is_parent = [max(coin_toss, 1 - coin_toss),
                                    min(coin_toss, 1 - coin_toss)]  # Proba regard objet pas parent / parent

        self.taste_is_good_age = self.generate_normal_distirbution(self.age - 2, 100, self.age, 5)
        self.taste_is_married_proba_look_opposite = 0 if not (self.is_married) else min(np.random.random(),
                                                                                        0.2)  # probabilite de regarde un objet d'un autre genre ou pas de son age

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

    def generate_normal_distirbution(self, min, max, center, std = None) :
        distrib = np.array(range(min, max))
        if std is None :
            sigma = np.std(distrib)
        else :
            sigma = std
        distribution = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(
            -(distrib - center) ** 2 / (2 * sigma ** 2) * 2)  # Suit loi normale
        return distribution

    def offer(self, item_recommended, is_list_reco = False) :

        coin_toss = np.random.random()
        if is_list_reco :
            raise Exception("Not defined")
        else :
            price, color, gender = item_recommended.get_properties
            proba_buy_price = self._taste[0][price]
            proba_buy_color = self._taste[1][color]
            interest = proba_buy_price * proba_buy_color
            return price if coin_toss < interest else interest
        # Ou faire P(A) + P(b) - P(A int B) ?

    # Offer mais cette fois tenant compte du gender, marrie, enfant... etc

    def offer_with_taste(self, item_recommended, is_list_reco = False) :

        if is_list_reco :
            raise Exception("Not defined")
        else :
            price, color, gender = item_recommended.get_properties
            # Proba vision
            for_parent, for_age, for_gender = item_recommended.get_audience
            proba_regard_parent = self.taste_is_parent[for_parent]
            if for_age < (self.age - 2) :
                proba_regard_age = 0
            else :
                proba_regard_age = self.taste_is_good_age[for_age - (self.age - 2)]
            proba_regard_gender = 1 if self.gender == for_gender else self.taste_is_married_proba_look_opposite
            coin_toss = np.random.random()
            if coin_toss > proba_regard_gender * proba_regard_age * proba_regard_parent :
                return 0  # Regarde pas

            # Proba achat
            coin_toss = np.random.random()
            proba_buy_price = self._taste[0][price]
            proba_buy_color = self._taste[1][color]
            interest = proba_buy_price * proba_buy_color
            return price if coin_toss < interest else interest

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

    @property
    def get_caracteristics(self) :
        return self.caracteristics
