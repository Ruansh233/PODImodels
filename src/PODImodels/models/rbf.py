import numpy as np
from typing import Optional

from scipy.interpolate import RBFInterpolator
from sklearn.linear_model import Ridge

from ..base import PODImodelAbstract


class fieldsRBF(PODImodelAbstract):
    def __init__(
        self,
        kernel: str = "linear",
        epsilon: float = 1.0,
        neighbors: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel: str = kernel
        self.epsilon: float = epsilon
        self.neighbors: Optional[int] = neighbors

    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        self.rbf: RBFInterpolator = RBFInterpolator(
            x, y, kernel=self.kernel, epsilon=self.epsilon, neighbors=self.neighbors
        )

    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_predict_input(x)
        return self._finalize_prediction(self.rbf(x), reconstruct_pod=False)


class PODRBF(PODImodelAbstract):
    uses_pod: bool = True

    def __init__(
        self,
        kernel: str = "linear",
        epsilon: float = 1.0,
        neighbors: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel: str = kernel
        self.epsilon: float = epsilon
        self.neighbors: Optional[int] = neighbors

    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        self.rbf: RBFInterpolator = RBFInterpolator(
            x, y, kernel=self.kernel, epsilon=self.epsilon, neighbors=self.neighbors
        )

    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_predict_input(x)
        return self._finalize_prediction(self.rbf(x), reconstruct_pod=True)


class fieldsRidgeRBF(PODImodelAbstract):
    def __init__(
        self,
        kernel: str = "linear",
        epsilon: float = 1.0,
        neighbors: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel: str = kernel
        self.epsilon: float = epsilon
        self.neighbors: Optional[int] = neighbors
        self.lin: Ridge = Ridge()

    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        self.lin.fit(x, y)
        self.rbf: RBFInterpolator = RBFInterpolator(
            x,
            y - self.lin.predict(x),
            kernel=self.kernel,
            epsilon=self.epsilon,
            neighbors=self.neighbors,
        )

    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_predict_input(x)
        y_pred = self.rbf(x) + self.lin.predict(x)
        return self._finalize_prediction(y_pred, reconstruct_pod=False)


class PODRidgeRBF(PODImodelAbstract):
    uses_pod: bool = True

    def __init__(
        self,
        kernel: str = "linear",
        epsilon: float = 1.0,
        neighbors: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel: str = kernel
        self.epsilon: float = epsilon
        self.neighbors: Optional[int] = neighbors
        self.lin: Ridge = Ridge()

    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        self.lin.fit(x, y)
        self.rbf: RBFInterpolator = RBFInterpolator(
            x,
            y - self.lin.predict(x),
            kernel=self.kernel,
            epsilon=self.epsilon,
            neighbors=self.neighbors,
        )

    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_predict_input(x)
        y_pred = self.lin.predict(x) + self.rbf(x)
        return self._finalize_prediction(y_pred, reconstruct_pod=True)
