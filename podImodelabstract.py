"""Module for the PODImodelAbstract abstract class"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pyvista as pv
from PODdata import vtk_writer


class PODImodelAbstract(ABC):
    """
    The abstract `PODImodelAbstract` class.

    All the classes that implement the input-output mapping should be inherited
    from this class.
    """

    @abstractmethod
    def fit(self, x, y):
        """Abstract `fit`"""

    @abstractmethod
    def predict(self, new_x):
        """Abstract `predict`"""

    def frobenius_norm(self, x, y, separate_err=False):
        if separate_err:
            err = []
            y_pred = self.predict(x)
            for i in range(len(x)):
                err.append(np.linalg.norm(y[i] - y_pred[i]) / 
                                          np.linalg.norm(y[i]))
            return np.array(err)
        else:
            return np.linalg.norm(y - self.predict(x)) / np.linalg.norm(y)
    
    def inf_norm(self, x, y, separate_err=False):
        if separate_err:
            err = []
            y_pred = self.predict(x)
            for i in range(len(x)):
                err.append(np.linalg.norm(y[i] - y_pred[i], ord=np.inf))
            return np.array(err)
        else:
            return np.linalg.norm(y - self.predict(x), ord=np.inf)
    
    def validate(self, x, y, training_ratio=0.8, rand_seed=42, norm="Frobenius"):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=training_ratio, random_state=rand_seed
        )
        self.fit(x_train, y_train)

        if norm == "Frobenius":
            return self.frobenius_norm(x_test, y_test)
        elif norm == "inf":
            return self.inf_norm(x_test, y_test)
        else:
            print("Please enter variable norm with value 'Frobenius' or 'inf'")
            assert False

    def reconstruct(self, x, y, refVTMName, saveFileName, dataType, is2D=False):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=0.8, random_state=42
        )
        self.fit(x_train, y_train)

        # Write the velocity data into VTK file
        refVTM = pv.MultiBlock(refVTMName)
        field_name = (
            [f"true_{i}" for i in range(x_test.shape[0])]
            + [f"rec_{i}" for i in range(x_test.shape[0])]
            + [f"err_{i}" for i in range(x_test.shape[0])]
        )

        # loop all test data and write the data into VTK file
        vtk_writer(
            np.vstack((y_test, self.predict(x_test), y_test - self.predict(x_test))),
            field_name,
            dataType,
            refVTM,
            saveFileName,
            is2D=is2D,
        )

    def fixed_validate(self, x_train, y_train, x_test, y_test, norm="Frobenius",
                       separate_err=False):
        self.fit(x_train, y_train)

        if norm == "Frobenius":
            return self.frobenius_norm(x_test, y_test, separate_err)
        elif norm == "inf":
            return self.inf_norm(x_test, y_test, separate_err)
        else:
            print("Please enter variable norm with value 'Frobenius' or 'inf'")
            assert False
