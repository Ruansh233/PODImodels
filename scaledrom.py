import numpy as np
from sklearn.model_selection import train_test_split


class maxMinScalar:
    def __init__(self, data):
        self.data = data
        self.min = np.min(data)
        self.max = np.max(data)

    def fit_transform(self):
        return (self.data - self.min) / (self.max - self.min)

    def transform(self, x):
        return (x - self.min) / (self.max - self.min)

    def inverse_transform(self, x):
        return x * (self.max - self.min) + self.min


class scaledROM:
    def __init__(self, x, y, scalar, ROM, train_size=0.8):
        self.x = x
        self.y = y
        self.scalar = scalar
        self.ROM = ROM
        self.train_size = train_size

    def scale_data(self, y):
        self.tmpscalar = self.scalar(y)
        return self.tmpscalar.fit_transform()

    def data_split(self, x, y, train_size):
        return train_test_split(x, y, train_size=train_size, random_state=42)

    def validate(self):
        data = self.scale_data(self.y)
        params_train, params_test, data_train, data_test = self.data_split(
            self.x, data, self.train_size
        )
        tmpROM = self.ROM
        tmpROM.fit(params_train, data_train)
        return np.linalg.norm(
            self.tmpscalar.inverse_transform(data_test)
            - self.tmpscalar.inverse_transform(tmpROM.predict(params_test))
        ) / np.linalg.norm(self.tmpscalar.inverse_transform(data_test))
