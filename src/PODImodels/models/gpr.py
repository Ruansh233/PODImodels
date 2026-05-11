import numpy as np
from typing import Optional

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF
from sklearn.linear_model import Ridge

from ..base import PODImodelAbstract


class fieldsGPR(PODImodelAbstract):
    def __init__(
        self, kernel: Optional[Kernel] = None, alpha: float = 1.0e-10, **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel: Kernel = (
            RBF(length_scale=1.0e0, length_scale_bounds="fixed")
            if kernel is None
            else kernel
        )
        self.alpha: float = alpha

    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        self.gpr: GaussianProcessRegressor = GaussianProcessRegressor(
            kernel=self.kernel, alpha=self.alpha
        )
        self.gpr.fit(x, y)

    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_predict_input(x)
        return self._finalize_prediction(self.gpr.predict(x), reconstruct_pod=False)


class PODGPR(PODImodelAbstract):
    uses_pod: bool = True

    def __init__(
        self, kernel: Optional[Kernel] = None, alpha: float = 1.0e-10, **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel: Kernel = (
            RBF(length_scale=1.0e0, length_scale_bounds="fixed")
            if kernel is None
            else kernel
        )
        self.alpha: float = alpha

    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        self.gpr: GaussianProcessRegressor = GaussianProcessRegressor(
            kernel=self.kernel, alpha=self.alpha
        )
        self.gpr.fit(x, y)

    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_predict_input(x)
        return self._finalize_prediction(self.gpr.predict(x), reconstruct_pod=True)


class fieldsRidgeGPR(PODImodelAbstract):
    def __init__(
        self, kernel: Optional[Kernel] = None, alpha: float = 1.0e-10, **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel: Kernel = (
            RBF(length_scale=1.0e0, length_scale_bounds="fixed")
            if kernel is None
            else kernel
        )
        self.alpha: float = alpha

    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        self.gpr: GaussianProcessRegressor = GaussianProcessRegressor(
            kernel=self.kernel, alpha=self.alpha
        )
        self.lin: Ridge = Ridge()
        self.lin.fit(x, y)
        self.gpr.fit(x, y - self.lin.predict(x))

    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_predict_input(x)
        y_pred = self.gpr.predict(x) + self.lin.predict(x)
        return self._finalize_prediction(y_pred, reconstruct_pod=False)


class PODRidgeGPR(PODImodelAbstract):
    uses_pod: bool = True

    def __init__(
        self, kernel: Optional[Kernel] = None, alpha: float = 1.0e-10, **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel: Kernel = (
            RBF(length_scale=1.0e0, length_scale_bounds="fixed")
            if kernel is None
            else kernel
        )
        self.alpha: float = alpha

    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        self.gpr: GaussianProcessRegressor = GaussianProcessRegressor(
            kernel=self.kernel, alpha=self.alpha
        )
        self.lin: Ridge = Ridge()
        self.lin.fit(x, y)
        self.gpr.fit(x, y - self.lin.predict(x))

    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        x = self._prepare_predict_input(x)
        y_pred = self.lin.predict(x) + self.gpr.predict(x)
        return self._finalize_prediction(y_pred, reconstruct_pod=True)
