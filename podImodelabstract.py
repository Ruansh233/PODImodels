"""Module for the PODImodelAbstract abstract class"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pyvista as pv
from PODdata import vtk_writer
from scipy.linalg import svd


class PODImodelAbstract(ABC):
    """
    The abstract `PODImodelAbstract` class.

    All the classes that implement the input-output mapping should be inherited
    from this class.
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        """Abstract `fit`"""

    @abstractmethod
    def predict(self, new_x: np.ndarray) -> np.ndarray:
        """Abstract `predict`"""

    def frobenius_norm(
        self, x: np.ndarray, y: np.ndarray, separate_err=False
    ) -> np.ndarray:
        """        Calculate the Frobenius norm of the difference between true and predicted values.
        If `separate_err` is True, return the error for each sample separately.
        Otherwise, return the overall error.
        Args:
            x (np.ndarray): Input features.
            y (np.ndarray): True target values.
            separate_err (bool): If True, return the error for each sample separately.
        Returns:
            np.ndarray: The Frobenius norm of the prediction error.
        """
        if separate_err:
            err = []
            y_pred = self.predict(x)
            for i in range(len(x)):
                err.append(np.linalg.norm(y[i] - y_pred[i]) / np.linalg.norm(y[i]))
            return np.array(err)
        else:
            return np.linalg.norm(y - self.predict(x)) / np.linalg.norm(y)

    def inf_norm(self, x: np.ndarray, y: np.ndarray, separate_err=False) -> np.ndarray:
        if separate_err:
            err = []
            y_pred = self.predict(x)
            for i in range(len(x)):
                err.append(np.linalg.norm(y[i] - y_pred[i], ord=np.inf))
            return np.array(err)
        else:
            return np.linalg.norm(y - self.predict(x), ord=np.inf)

    def performPOD(self, y: np.ndarray) -> np.ndarray:
        """
        Perform Proper Orthogonal Decomposition (POD) on the training data.
        This method is called in the `fit` method of the derived classes.
        Args:
            y (np.ndarray): The training data for which POD is to be performed.
        Returns:
            np.ndarray: The coefficients of the POD modes.
        Raises:
            ValueError: If the rank is greater than the number of modes.
        """
        # perform POD reduction if not already done
        if not hasattr(self, "v_all"):
            self.reduction(y)
            print("POD reduction completed.")
        if self.rank > self.v_all.shape[0]:
            raise ValueError("Rank is greater than the number of modes.")
        self.s = self.s_all[: self.rank]
        self.v = self.v_all[: self.rank]
        return self.coeffs[:, : self.rank]

    def reduction(self, y):
        """
        Perform Proper Orthogonal Decomposition (POD) on the training data.
        This method is called in the `fit` method of the derived classes.
        Args:
            y (np.ndarray): The training data for which POD is to be performed.
        Returns:
            np.ndarray: The coefficients of the POD modes.
        """
        u, self.s_all, self.v_all = svd(y, full_matrices=False)
        self.coeffs = u @ np.diag(self.s_all)

    def validate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        training_ratio: float = 0.8,
        rand_seed: int = 42,
        norm: str = "Frobenius",
    ):
        """
        Validate the model using a train-test split.
        Args:
            x (np.ndarray): Input features.
            y (np.ndarray): Target values.
            training_ratio (float): Ratio of the training set.
            rand_seed (int): Random seed for reproducibility.
            norm (str): Type of norm to use for validation ('Frobenius' or 'inf').
        Returns:
            float: The calculated norm of the prediction error.
        """
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

    def reconstruct(
        self,
        x: np.ndarray,
        y: np.ndarray,
        refVTMName: str,
        saveFileName: str,
        dataType: str,
        is2D: bool = False,
    ):
        """
        Reconstruct the model and write the results into a VTK file.
        Args:
            x (np.ndarray): Input features.
            y (np.ndarray): Target values.
            refVTMName (str): Name of the reference VTM file.
            saveFileName (str): Name of the file to save the results.
            dataType (str): Type of data to be written.
            is2D (bool): Whether the data is 2D or not.
        """
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

    def fixed_validate(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        norm: str = "Frobenius",
        separate_err: bool = False,
    ):
        """
        Validate the model with fixed training and testing data.
        Args:
            x_train (np.ndarray): Training input features.
            y_train (np.ndarray): Training target values.
            x_test (np.ndarray): Testing input features.
            y_test (np.ndarray): Testing target values. 
            norm (str): Type of norm to use for validation ('Frobenius' or 'inf').
            separate_err (bool): If True, return the error for each sample separately.
        Returns:
            float: The calculated norm of the prediction error.
        """
        self.fit(x_train, y_train)

        if norm == "Frobenius":
            return self.frobenius_norm(x_test, y_test, separate_err)
        elif norm == "inf":
            return self.inf_norm(x_test, y_test, separate_err)
        else:
            print("Please enter variable norm with value 'Frobenius' or 'inf'")
            assert False

    def multi_validate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ranks: list,
        training_ratio: float = 0.8,
        rand_seed: int = 42,
        norm: str = "Frobenius",
    ):
        """
        Validate the model with multiple ranks.
        Args:
            x (np.ndarray): Input features.
            y (np.ndarray): Target values.
            ranks (list): List of ranks to validate.
            training_ratio (float): Ratio of the training set.
            rand_seed (int): Random seed for reproducibility.
            norm (str): Type of norm to use for validation ('Frobenius' or 'inf').
        Returns:
            np.ndarray: Array of errors for each rank.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=training_ratio, random_state=rand_seed
        )
        self.errors = []
        for i in ranks:
            self.rank = i
            self.fit(x_train, y_train)
            if norm == "Frobenius":
                self.errors.append(self.frobenius_norm(x_test, y_test))
            elif norm == "inf":
                self.errors.append(self.inf_norm(x_test, y_test))
            else:
                print("Please enter variable norm with value 'Frobenius' or 'inf'")
                assert False
        return np.array(self.errors)

    def multi_validate_fixed(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        ranks: list,
        norm: str = "Frobenius",
        separate_err: bool = False,
    ):
        """
        Validate the model with multiple ranks using fixed training and testing data.
        Args:
            x_train (np.ndarray): Training input features.
            y_train (np.ndarray): Training target values.
            x_test (np.ndarray): Testing input features.
            y_test (np.ndarray): Testing target values.
            ranks (list): List of ranks to validate.
            norm (str): Type of norm to use for validation ('Frobenius' or 'inf').
            separate_err (bool): If True, return the error for each sample separately.
        Returns:
            np.ndarray: Array of errors for each rank.
        """
        self.errors = []
        for i in ranks:
            self.rank = i
            self.fit(x_train, y_train)
            if norm == "Frobenius":
                self.errors.append(self.frobenius_norm(x_test, y_test, separate_err))
            elif norm == "inf":
                self.errors.append(self.inf_norm(x_test, y_test, separate_err))
            else:
                print("Please enter variable norm with value 'Frobenius' or 'inf'")
                assert False
        return np.array(self.errors)
