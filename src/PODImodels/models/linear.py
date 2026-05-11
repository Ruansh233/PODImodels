import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from ..base import PODImodelAbstract


class fieldsLinear(PODImodelAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lin: LinearRegression = LinearRegression()

    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        self.lin.fit(x, y)

    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_predict_input(x)
        return self._finalize_prediction(self.lin.predict(x), reconstruct_pod=False)


class PODLinear(PODImodelAbstract):
    uses_pod: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lin: LinearRegression = LinearRegression()

    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        self.lin.fit(x, y)

    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_predict_input(x)
        return self._finalize_prediction(self.lin.predict(x), reconstruct_pod=True)


class fieldsRidge(PODImodelAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lin: Ridge = Ridge()

    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        self.lin.fit(x, y)

    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_predict_input(x)
        return self._finalize_prediction(self.lin.predict(x), reconstruct_pod=False)


class PODRidge(PODImodelAbstract):
    uses_pod: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lin: Ridge = Ridge()

    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        self.lin.fit(x, y)

    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_predict_input(x)
        return self._finalize_prediction(self.lin.predict(x), reconstruct_pod=True)
