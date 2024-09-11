"""Module for the PODImodelAbstract abstract class"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class PODImodelAbstract(ABC):
    """
    The abstract `PODImodelAbstract` class.

    All the classes that implement the input-output mapping should be inherited
    from this class.
    """

    @abstractmethod
    def fit(self, points, values):
        """Abstract `fit`"""

    @abstractmethod
    def predict(self, new_point):
        """Abstract `predict`"""

    def validate(self, x, y, training_ratio=0.8, rand_seed=42):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=training_ratio, random_state=rand_seed
        )
        self.fit(x_train, y_train)
        return np.linalg.norm(y_test - self.predict(x_test)) / np.linalg.norm(y_test)
