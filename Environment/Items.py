import numpy as np
np.random.seed(42)

class Items:
    def __init__(self,id, interval_price = 100, interval_color = 5):
        self._id = id
        self._price = np.random.randint(interval_price)
        self._color = np.random.randint(interval_color)
        self._for_women = np.random.randint(2)

    def list_item(self):
        return {"Price " : self._price, "Color ": self._color, "Women item" : self._for_women}

    def get_as_list(self):
        return [self._id, self._price, self._color, self._for_women]

    @property
    def get_properties(self):
        return self._price, self._color, self._for_women
    @property
    def get_id(self):
        return self._id