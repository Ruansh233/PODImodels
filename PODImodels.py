import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.linear_model import Ridge
from scipy.linalg import svd
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from podImodelabstract import PODImodelAbstract


class fieldsRidge(PODImodelAbstract):
    def __init__(self):
        self.lin = Ridge()

    def fit(self, x, y):
        self.lin.fit(x, y)

    def predict(self, x):
        return self.lin.predict(x)


class PODRidge(PODImodelAbstract):
    def __init__(self, rank=10, with_scalar=True):
        self.lin = Ridge()
        self.rank = rank
        self.with_scalar = with_scalar

    def fit(self, x, y):
        v = svd(y, full_matrices=False)[2]
        self.v = v[: self.rank]
        y = y @ self.v.T

        if self.with_scalar:        
            self.coeffs_scalar = MinMaxScaler()
            y = self.coeffs_scalar.fit_transform(y)

        self.lin.fit(x, y)

    def predict(self, x):
        if self.with_scalar:
            return self.coeffs_scalar.inverse_transform(self.lin.predict(x)) @ self.v
        else:
            return self.lin.predict(x) @ self.v


class fieldsGPR(PODImodelAbstract):
    def __init__(self, kernel=None, alpha=1.0e-10):
        if kernel is None:
            self.kernel = RBF(length_scale=1.0e0, length_scale_bounds="fixed")
        else:
            self.kernel = kernel
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=alpha)

    def fit(self, x, y):
        self.gpr.fit(x, y)

    def predict(self, x):
        return self.gpr.predict(x)


class PODGPR(PODImodelAbstract):
    def __init__(self, kernel=None, alpha=1.0e-10, rank=10, with_scalar=True):
        if kernel is None:
            self.kernel = RBF(length_scale=1.0e0, length_scale_bounds="fixed")
        else:
            self.kernel = kernel
        self.alpha = alpha
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
        self.rank = rank
        self.with_scalar = with_scalar

    def fit(self, x, y):
        v = svd(y, full_matrices=False)[2]
        self.v = v[: self.rank]
        y = y @ self.v.T

        if self.with_scalar:
            self.coeffs_scalar = MinMaxScaler()
            y = self.coeffs_scalar.fit_transform(y)

        self.gpr.fit(x, y)

    def predict(self, x):
        if self.with_scalar:
            return self.coeffs_scalar.inverse_transform(self.gpr.predict(x)) @ self.v
        else:
            return self.gpr.predict(x) @ self.v


class fieldsRidgeGPR(PODImodelAbstract):
    def __init__(self, kernel=None, alpha=1.0e-10):
        if kernel is None:
            self.kernel = RBF(length_scale=1.0e0, length_scale_bounds="fixed")
        else:
            self.kernel = kernel
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=alpha)
        self.lin = Ridge()

    def fit(self, x, y):
        self.lin.fit(x, y)
        self.gpr.fit(x, y - self.lin.predict(x))

    def predict(self, x):
        return self.gpr.predict(x) + self.lin.predict(x)


class PODRidgeGPR(PODImodelAbstract):
    def __init__(self, kernel=None, alpha=1.0e-10, rank=10, with_scalar=True):
        if kernel is None:
            self.kernel = RBF(length_scale=1.0e0, length_scale_bounds="fixed")
        else:
            self.kernel = kernel
        self.alpha = alpha
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
        self.lin = Ridge()
        self.rank = rank
        self.with_scalar = with_scalar

    def fit(self, x, y):
        v = svd(y, full_matrices=False)[2]
        self.v = v[: self.rank]
        y = y @ self.v.T

        if self.with_scalar:
            self.coeffs_scalar = MinMaxScaler()
            y = self.coeffs_scalar.fit_transform(y)

        self.lin.fit(x, y)
        self.gpr.fit(x, y - self.lin.predict(x))

    def predict(self, x):
        if self.with_scalar:
            tmp = self.coeffs_scalar.inverse_transform(
                self.lin.predict(x) + self.gpr.predict(x)
            )
            return tmp @ self.v
        else:
            return (self.lin.predict(x) + self.gpr.predict(x)) @ self.v


class fieldsRBF(PODImodelAbstract):
    def __init__(self, kernel="linear", epsilon=1.0):
        self.kernel = kernel
        self.epsilon = epsilon

    def fit(self, x, y):
        self.rbf = RBFInterpolator(x, y, kernel=self.kernel, epsilon=self.epsilon)

    def predict(self, x):
        return self.rbf(x)


class PODRBF(PODImodelAbstract):
    def __init__(self, kernel="linear", epsilon=1.0, rank=10, with_scalar=True):
        self.kernel = kernel
        self.epsilon = epsilon
        self.rank = rank
        self.with_scalar = with_scalar

    def fit(self, x, y):
        v = svd(y, full_matrices=False)[2]
        self.v = v[: self.rank]
        y = y @ self.v.T

        if self.with_scalar:
            self.coeffs_scalar = MinMaxScaler()
            y = self.coeffs_scalar.fit_transform(y)

        self.rbf = RBFInterpolator(x, y, kernel=self.kernel, epsilon=self.epsilon)

    def predict(self, x):
        if self.with_scalar:
            tmp = self.coeffs_scalar.inverse_transform(self.rbf(x))
            return tmp @ self.v
        else:
            return self.rbf(x) @ self.v


class fieldsRidgeRBF(PODImodelAbstract):
    def __init__(self, kernel="linear", epsilon=1.0):
        self.kernel = kernel
        self.epsilon = epsilon
        self.lin = Ridge()

    def fit(self, x, y):
        self.lin.fit(x, y)
        self.rbf = RBFInterpolator(
            x, y - self.lin.predict(x), kernel=self.kernel, epsilon=self.epsilon
        )

    def predict(self, x):
        return self.rbf(x) + self.lin.predict(x)


class PODRidgeRBF(PODImodelAbstract):
    def __init__(self, kernel="linear", epsilon=1.0, rank=10, with_scalar=True):
        self.kernel = kernel
        self.epsilon = epsilon
        self.lin = Ridge()
        self.rank = rank
        self.with_scalar = with_scalar

    def fit(self, x, y):
        v = svd(y, full_matrices=False)[2]
        self.v = v[: self.rank]
        y = y @ self.v.T

        if self.with_scalar:
            self.coeffs_scalar = MinMaxScaler()
            y = self.coeffs_scalar.fit_transform(y)

        self.lin.fit(x, y)
        self.rbf = RBFInterpolator(
            x, y - self.lin.predict(x), kernel=self.kernel, epsilon=self.epsilon
        )

    def predict(self, x):
        if self.with_scalar:
            tmp = self.coeffs_scalar.inverse_transform(
                self.lin.predict(x) + self.rbf(x)
            )
            return tmp @ self.v
        else:
            return (self.lin.predict(x) + self.rbf(x)) @ self.v


# class testPODRBF(PODImodelAbstract):
#     def __init__(self, kernel="linear", epsilon=1.0, rank=10, with_scalar=True):
#         self.kernel = kernel
#         self.epsilon = epsilon
#         self.rank = rank
#         self.with_scalar = with_scalar
#         self.x_scalar = MinMaxScaler()
#         self.y_scalar = MinMaxScaler()

#     def fit(self, x, y):
#         u, s, v = svd(y, full_matrices=False)
#         self.v = v[: self.rank]
#         y = y @ self.v.T

#         if self.with_scalar:
#             x = self.x_scalar.fit_transform(x)
#             y = self.y_scalar.fit_transform(y)

#         self.rbf = RBFInterpolator(x, y, kernel=self.kernel, epsilon=self.epsilon)

#     def predict(self, x):
#         if self.with_scalar:
#             x = self.x_scalar.transform(x)
#             tmp = self.y_scalar.inverse_transform(self.rbf(x))
#             return tmp @ self.v
#         else:
#             return self.rbf(x) @ self.v


# class testPODRidgeGPR(PODImodelAbstract):
#     def __init__(self, kernel=None, alpha=1.0e-10, rank=10, with_scalar=True):
#         if kernel is None:
#             self.kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
#                 1.0, length_scale_bounds="fixed"
#             )
#         else:
#             self.kernel = kernel
#         self.alpha = alpha
#         self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
#         self.lin = Ridge()
#         self.rank = rank
#         self.with_scalar = with_scalar

#     def fit(self, x, y):
#         u, s, v = svd(y, full_matrices=False)
#         self.v = v[: self.rank]
#         y = y @ self.v.T

#         if self.with_scalar:
#             # self.x_min = np.min(x)
#             # self.x_max = np.max(x)
#             self.y_min = np.min(y)
#             self.y_max = np.max(y)

#             # x = (x - self.x_min)/(self.x_max - self.x_min)
#             y = (y - self.y_min) / (self.y_max - self.y_min)

#             # print(f"self.y_min, {self.y_min}")

#         self.lin.fit(x, y)
#         self.gpr.fit(x, y - self.lin.predict(x))

#     def predict(self, x):
#         if self.with_scalar:
#             # x = (x - self.x_min)/(self.x_max - self.x_min)
#             tmp = self.lin.predict(x) + self.gpr.predict(x)
#             tmp = tmp * (self.y_max - self.y_min) + self.y_min
#             return tmp @ self.v
#         else:
#             return (self.lin.predict(x) + self.gpr.predict(x)) @ self.v


# class testPODRidgeGPR2(PODImodelAbstract):
#     def __init__(self, kernel=None, alpha=1.0e-10, rank=10, with_scalar=True):
#         if kernel is None:
#             self.kernel = RBF(length_scale=1.0e0, length_scale_bounds="fixed")
#         else:
#             self.kernel = kernel
#         self.alpha = alpha
#         self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
#         self.lin = Ridge()
#         self.rank = rank
#         self.with_scalar = with_scalar
#         # if self.with_scalar:
#         #     self.x_scalar = MinMaxScaler()
#         #     self.y_scalar = MinMaxScaler()

#     def fit(self, x, y):
#         if self.with_scalar:
#             self.x_min = np.min(x)
#             self.x_max = np.max(x)
#             self.y_min = np.min(y)
#             self.y_max = np.max(y)

#             # print("self.y_min ", self.y_min)
#             # print("self.y_max", self.y_max)

#             # x = (x - self.x_min) / (self.x_max - self.x_min)
#             y = (y - self.y_min) / (self.y_max - self.y_min)

#         v = svd(y, full_matrices=False)[2]
#         self.v = v[: self.rank]
#         y = y @ self.v.T

#         # self.coeffs_scalar = MinMaxScaler()
#         # y = self.coeffs_scalar.fit_transform(y)
#         self.coeffs_min = np.min(y)
#         self.coeffs_max = np.max(y)
#         y = (y - self.coeffs_min) / (self.coeffs_max - self.coeffs_min)

#         self.lin.fit(x, y)
#         self.gpr.fit(x, y - self.lin.predict(x))

#     def predict(self, x):
#         if self.with_scalar:
#             # x = (x - self.x_min) / (self.x_max - self.x_min)
#             tmp = self.lin.predict(x) + self.gpr.predict(x)
#             tmp = tmp * (self.coeffs_max - self.coeffs_min) + self.coeffs_min
#             return (tmp @ self.v) * (self.y_max - self.y_min) + self.y_min
#         else:
#             return (
#                 (self.lin.predict(x) + self.gpr.predict(x))
#                 * (self.coeffs_max - self.coeffs_min)
#                 + self.coeffs_min
#             ) @ self.v
