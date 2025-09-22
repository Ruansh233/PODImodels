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
    def __init__(self, scalar, ROM):
        self.scalar = scalar
        self.ROM = ROM

    def scale_data(self, y):
        self.tmpscalar = self.scalar(y)
        return self.tmpscalar.fit_transform()

    def data_split(self, x, y, train_size):
        return train_test_split(x, y, train_size=train_size, random_state=42)

    def fit(self, x, y):
        data = self.scale_data(y)
        self.ROM.fit(x, data)

    def predict(self, x):
        return self.ROM.predict(x)

    def validate(self, x, y, training_ratio=0.8, rand_seed=42, norm="Frobenius"):
        y = self.scale_data(y)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=training_ratio, random_state=rand_seed
        )

        self.fit(x_train, y_train)

        if norm == "Frobenius":
            return np.linalg.norm(
                self.tmpscalar.inverse_transform(y_test)
                - self.tmpscalar.inverse_transform(self.predict(x_test))
            ) / np.linalg.norm(self.tmpscalar.inverse_transform(y_test))
        elif norm == "inf":
            return np.max(np.abs(y_test - self.predict(x_test)))
        else:
            print("Please enter variable norm with value 'Frobenius' or 'inf'")
            assert False
