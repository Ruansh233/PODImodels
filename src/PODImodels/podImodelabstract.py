"""
POD-based Interpolation Model Abstract Base Class
=================================================

This module defines the abstract base class for all POD-based interpolation models.
It provides a common interface and shared functionality for building reduced-order
models that combine Proper Orthogonal Decomposition with various machine learning
techniques.

Classes
-------
PODImodelAbstract
    Abstract base class for POD-based interpolation models.
"""

from abc import ABC, abstractmethod
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pyvista as pv
from .PODdata import vtk_writer
from scipy.linalg import svd
from typing import Optional, Union, Tuple, List


class PODImodelAbstract(ABC):
    """
    Abstract base class for POD-based interpolation models.

    This class provides a common interface for building reduced-order models that
    combine Proper Orthogonal Decomposition (POD) with various machine learning
    techniques. All concrete interpolation model classes should inherit from this
    class and implement the abstract methods.

    The class handles common functionality including:
    - Data scaling and preprocessing
    - POD decomposition for dimensionality reduction
    - Model validation and error assessment
    - VTK file output for visualization

    Parameters
    ----------
    rank : int, optional
        The number of POD modes to retain. Default is 10.
    with_scaler_x : bool, optional
        Whether to apply MinMax scaling to input features. Default is True.
    with_scaler_y : bool, optional
        Whether to apply MinMax scaling to target values. Default is True.
    POD_algo : {'svd', 'eigen'}, optional
        The algorithm to use for POD computation. 'svd' uses singular value
        decomposition, 'eigen' uses eigenvalue decomposition. Default is 'eigen'.

    Attributes
    ----------
    rank : int
        Number of POD modes to retain.
    with_scaler_x : bool
        Flag for input scaling.
    with_scaler_y : bool
        Flag for output scaling.
    POD_algo : str
        POD algorithm type.
    scalar_X : MinMaxScaler, optional
        Scaler for input features (created if with_scaler_x=True).
    scalar_Y : MinMaxScaler, optional
        Scaler for output features (created if with_scaler_y=True).
    v : np.ndarray
        Truncated POD modes matrix.
    v_all : np.ndarray
        Full POD modes matrix.
    s : np.ndarray
        Truncated singular values.
    s_all : np.ndarray
        Full singular values.
    coeffs : np.ndarray
        POD coefficients matrix.

    Notes
    -----
    Subclasses must implement `fit_tmp` and `predict_tmp` methods to define
    the specific machine learning algorithm used for interpolation.

    Examples
    --------
    >>> # Define a concrete implementation (example)
    >>> class MyPODModel(PODImodelAbstract):
    ...     def __init__(self, **kwargs):
    ...         super().__init__(**kwargs)
    ...         # Initialize specific model
    ...
    ...     def fit_tmp(self, x, y):
    ...         # Implement specific fitting logic
    ...         pass
    ...
    ...     def predict_tmp(self, x):
    ...         # Implement specific prediction logic
    ...         return predictions
    """

    @abstractmethod
    def __init__(
        self,
        rank: int = 10,
        with_scaler_x: bool = True,
        with_scaler_y: bool = True,
        POD_algo: str = "eigen",
    ):
        """
        Abstract initialization method.

        Parameters
        ----------
        rank : int, optional
            The number of POD modes to retain. Default is 10.
        with_scaler_x : bool, optional
            Whether to apply MinMax scaling to input features. Default is True.
        with_scaler_y : bool, optional
            Whether to apply MinMax scaling to target values. Default is True.
        POD_algo : {'svd', 'eigen'}, optional
            The algorithm to use for POD computation. Default is 'eigen'.
        """
        self.rank: int = rank
        self.with_scaler_x: bool = with_scaler_x
        self.with_scaler_y: bool = with_scaler_y
        self.POD_algo: str = POD_algo
        self.scalar_X: Optional[MinMaxScaler] = None
        self.scalar_Y: Optional[MinMaxScaler] = None
        self.v: Optional[np.ndarray] = None
        self.v_all: Optional[np.ndarray] = None
        self.s: Optional[np.ndarray] = None
        self.s_all: Optional[np.ndarray] = None
        self.coeffs: Optional[np.ndarray] = None

    @abstractmethod
    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Abstract method for model-specific fitting logic.

        This method should be implemented by subclasses to define the specific
        machine learning algorithm used for learning the input-output mapping.

        Parameters
        ----------
        x : np.ndarray
            Preprocessed input features of shape (n_samples, n_features).
        y : np.ndarray
            Preprocessed target values of shape (n_samples, n_targets).
        """

    @abstractmethod
    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        """
        Abstract method for model-specific prediction logic.

        This method should be implemented by subclasses to define how predictions
        are made using the trained model.

        Parameters
        ----------
        x : np.ndarray
            Input features for prediction.

        Returns
        -------
        np.ndarray
            Predicted values.
        """

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the training data.

        This method handles the complete training pipeline including input validation,
        POD decomposition (if applicable), data scaling, and calling the model-specific
        fitting method.

        Parameters
        ----------
        x : np.ndarray
            Input features of shape (n_samples, n_input_features).
        y : np.ndarray
            Target values of shape (n_samples, n_output_features).

        Raises
        ------
        ValueError
            If the number of samples in x and y don't match, or if inputs are not 2D.
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X_train and y_train must match.")
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("Input and output data must be 2D numpy arrays.")

        if "POD" in self.__class__.__name__:
            y = self.performPOD(y)

        if self.with_scaler_x:
            self.scalar_X = MinMaxScaler()
            x = self.scalar_X.fit_transform(x)
        if self.with_scaler_y:
            self.scalar_Y = MinMaxScaler()
            y = self.scalar_Y.fit_transform(y)

        self.fit_tmp(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict target values for given input features.

        This method serves as a wrapper around the model-specific prediction method,
        ensuring consistent interface across all model types.

        Parameters
        ----------
        x : np.ndarray
            Input features of shape (n_samples, n_input_features).

        Returns
        -------
        np.ndarray
            Predicted target values of shape (n_samples, n_output_features).
        """
        return self.predict_tmp(x)

    def frobenius_norm(
        self,
        x: np.ndarray,
        y: np.ndarray,
        separate_err: bool = False,
        lift_y: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate the Frobenius norm of prediction errors.

        Computes the relative Frobenius norm between true and predicted values,
        which provides a measure of the overall prediction accuracy.

        Parameters
        ----------
        x : np.ndarray
            Input features for prediction.
        y : np.ndarray
            True target values.
        separate_err : bool, optional
            If True, return the error for each sample separately.
            If False, return the overall aggregated error. Default is False.
        lift_y : np.ndarray, optional
            If provided, this array is added to both true and predicted values
            before error calculation. Default is None.

        Returns
        -------
        np.ndarray or float
            If separate_err=True, returns array of relative errors for each sample.
            If separate_err=False, returns overall relative Frobenius norm error.

        Notes
        -----
        The relative Frobenius norm is calculated as:
        ||y_true - y_pred||_F / ||y_true||_F

        For separate errors, it's calculated per sample as:
        ||y_true[i] - y_pred[i]||_2 / ||y_true[i]||_2
        """
        if separate_err:
            err = []
            y_pred = self.predict(x)
            if lift_y is not None:
                y_pred += lift_y
                y += lift_y
            for i in range(len(x)):
                err.append(np.linalg.norm(y[i] - y_pred[i]) / np.linalg.norm(y[i]))
            return np.array(err)
        else:
            if lift_y is not None:
                y_pred = self.predict(x) + lift_y
                y = y + lift_y
                return np.linalg.norm(y - y_pred) / np.linalg.norm(y)
            return np.linalg.norm(y - self.predict(x)) / np.linalg.norm(y)

    def inf_norm(
        self, x: np.ndarray, y: np.ndarray, separate_err: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Calculate the infinity norm of prediction errors.

        Computes the maximum absolute error between true and predicted values,
        which provides a measure of the worst-case prediction error.

        Parameters
        ----------
        x : np.ndarray
            Input features for prediction.
        y : np.ndarray
            True target values.
        separate_err : bool, optional
            If True, return the error for each sample separately.
            If False, return the overall aggregated error. Default is False.

        Returns
        -------
        np.ndarray or float
            If separate_err=True, returns array of infinity norm errors for each sample.
            If separate_err=False, returns overall infinity norm error.

        Notes
        -----
        The infinity norm is the maximum absolute difference:
        ||y_true - y_pred||_∞ = max|y_true - y_pred|
        """
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
        Perform Proper Orthogonal Decomposition on the training data.

        This method applies POD to reduce the dimensionality of the target data
        from the full field representation to a reduced set of POD coefficients.
        It handles the truncation to the specified rank and validates the input.

        Parameters
        ----------
        y : np.ndarray
            Training data matrix of shape (n_samples, n_features) for which
            POD decomposition is to be performed.

        Returns
        -------
        np.ndarray
            POD coefficients matrix of shape (n_samples, rank) representing
            the training data in the reduced POD space.

        Raises
        ------
        ValueError
            If the specified rank is greater than the number of available modes.

        Notes
        -----
        This method calls `reduction` if POD has not been computed yet, then
        truncates the modes and coefficients to the specified rank. The POD
        decomposition follows: y ≈ coeffs @ modes, where coeffs are the returned
        values and modes are stored in self.v.
        """
        if not hasattr(self, "v_all"):
            self.reduction(y)
        if self.rank > self.v_all.shape[0]:
            raise ValueError("Rank is greater than the number of modes.")
        self.s = self.s_all[: self.rank]
        self.v = self.v_all[: self.rank]
        return self.coeffs[:, : self.rank]

    def reduction(self, y: np.ndarray) -> None:
        """
        Perform POD using the specified algorithm (SVD or eigenvalue decomposition).

        This method computes the full POD decomposition of the training data using
        either singular value decomposition or eigenvalue decomposition, depending
        on the POD_algo parameter.

        Parameters
        ----------
        y : np.ndarray
            Training data matrix of shape (n_samples, n_features).

        Raises
        ------
        ValueError
            If an invalid POD algorithm is specified.

        Notes
        -----
        Two algorithms are supported:

        1. 'svd': Direct SVD decomposition
           - More accurate for well-conditioned problems
           - Better numerical stability
           - Recommended for most applications

        2. 'eigen': Eigenvalue decomposition of the covariance matrix
           - More memory efficient for wide matrices (n_features >> n_samples)
           - Potentially less stable for ill-conditioned problems
           - Useful when n_samples << n_features

        The method stores the full decomposition in attributes:
        - s_all: all singular values
        - v_all: all POD modes (right singular vectors)
        - coeffs: all POD coefficients (scaled left singular vectors)
        """
        if self.POD_algo == "svd":
            u, self.s_all, self.v_all = svd(y, full_matrices=False)
            self.coeffs = u @ np.diag(self.s_all)
            print(f"POD_SVD reduction completed.")
        elif self.POD_algo == "eigen":
            N, M = y.shape

            C = y @ y.T
            eigenvalues, U = np.linalg.eigh(C)

            sorted_indices = np.argsort(eigenvalues)[::-1]
            sorted_eigenvalues = eigenvalues[sorted_indices]
            U = U[:, sorted_indices]

            self.s_all = np.sqrt(sorted_eigenvalues)
            self.coeffs = U @ np.diag(self.s_all)

            self.v_all = np.zeros((N, M))
            tolerance = 1e-10
            for i in range(N):
                if self.s_all[i] > tolerance:
                    u_i = U[:, i]
                    self.v_all[i, :] = (1 / self.s_all[i]) * (u_i.T @ y)
            print("POD_eigen reduction completed.")
        else:
            raise ValueError("Invalid POD method.")

    def truncation_error(self) -> Tuple[float, float]:
        """
        Calculate the truncation error of the POD decomposition.

        This method computes the relative truncation error based on the singular
        values obtained from the POD decomposition. The truncation error quantifies
        the amount of information lost by retaining only a subset of the POD modes.

        Returns
        -------
        float
            The relative truncation error, defined as:
            (sum of discarded singular values) / (sum of all singular values).
        """
        if not hasattr(self, "s_all") or not hasattr(self, "s"):
            raise ValueError("POD decomposition has not been performed yet.")
        total_energy = np.sum(self.s_all**2)
        retained_energy = np.cumsum(self.s_all**2)
        truncation_error = 1 - retained_energy / total_energy
        projection_error = np.sqrt(truncation_error)

        return truncation_error, projection_error

    def reconstruct(
        self,
        x: np.ndarray,
        y: np.ndarray,
        refVTMName: str,
        saveFileName: str,
        dataType: str,
        x_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        is2D: bool = False,
    ) -> None:
        """
        Reconstruct the model predictions and save results to VTK files.

        This method trains the model and generates VTK files containing the true values,
        reconstructed values, and prediction errors for visualization and analysis.

        Parameters
        ----------
        x : np.ndarray
            Input features. Used for train-test split if specific splits not provided.
        y : np.ndarray
            Target values. Used for train-test split if specific splits not provided.
        refVTMName : str
            Path to the reference VTM file that provides the mesh structure.
        saveFileName : str
            Base filename for saving the reconstruction results.
        dataType : {'scalar', 'vector'}
            Type of data being reconstructed for VTK output.
        x_train : np.ndarray, optional
            Specific training input features. If None, automatic split is used.
        y_train : np.ndarray, optional
            Specific training target values. If None, automatic split is used.
        x_test : np.ndarray, optional
            Specific testing input features. If None, automatic split is used.
        y_test : np.ndarray, optional
            Specific testing target values. If None, automatic split is used.
        is2D : bool, optional
            Whether the data is 2D (for vector fields). Default is False.

        Notes
        -----
        The method creates a VTK file with three sets of fields for each test sample:
        - 'true_{i}': Original target values
        - 'rec_{i}': Reconstructed/predicted values
        - 'err_{i}': Absolute error (true - predicted)

        If no specific train/test split is provided, the method uses an 80-20 split
        with random_state=42.

        Examples
        --------
        >>> model.reconstruct(X, Y, 'mesh.vtm', 'results', 'vector', is2D=True)
        # Creates results.vtm with true, reconstructed, and error fields
        """
        if x_train is None or y_train is None or x_test is None or y_test is None:
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

    def validate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        training_ratio: float = 0.8,
        rand_seed: int = 42,
        norm: str = "Frobenius",
        separate_err: bool = False,
        lift_y: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """
        Validate the model using a train-test split.

        This method provides a convenient way to assess model performance by
        automatically splitting the data, training the model, and computing
        prediction errors on the test set.

        Parameters
        ----------
        x : np.ndarray
            Input features of shape (n_samples, n_input_features).
        y : np.ndarray
            Target values of shape (n_samples, n_output_features).
        training_ratio : float, optional
            Fraction of data to use for training (0 < training_ratio < 1).
            Default is 0.8.
        rand_seed : int, optional
            Random seed for reproducible train-test splits. Default is 42.
        norm : {'Frobenius', 'inf'}, optional
            Type of norm to use for error calculation. Default is 'Frobenius'.
        separate_err : bool, optional
            If True, return the error for each test sample separately.
            If False, return the overall aggregated error. Default is False.
        lift_y : np.ndarray, optional
            If provided, this array is added to both true and predicted values
            before error calculation. Default is None.

        Returns
        -------
        float or np.ndarray
            The calculated norm of the prediction error on the test set.
            If separate_err=True, returns an array of errors for each test sample.
            If separate_err=False, returns a single aggregated error value.

        Raises
        ------
        AssertionError
            If an invalid norm type is specified.

        Examples
        --------
        >>> model = SomeConcreteModel(rank=10)
        >>> error = model.validate(X, Y, training_ratio=0.7, norm='Frobenius')
        >>> print(f"Validation error: {error:.6f}")
        >>> # Using lifting to adjust predictions
        >>> lift = np.mean(Y, axis=0)
        >>> error_lifted = model.validate(X, Y, lift_y=lift, norm='inf')
        >>> print(f"Validation error with lifting: {error_lifted:.6f}")
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=training_ratio, random_state=rand_seed
        )
        self.fit(x_train, y_train)

        if norm == "Frobenius":
            return self.frobenius_norm(
                x_test, y_test, lift_y=lift_y, separate_err=separate_err
            )
        elif norm == "inf":
            return self.inf_norm(x_test, y_test, separate_err=separate_err)
        else:
            print("Please enter variable norm with value 'Frobenius' or 'inf'")
            assert False

    def fixed_validate(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        norm: str = "Frobenius",
        separate_err: bool = False,
        lift_y: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """
        Validate the model with fixed training and testing datasets.

        This method allows for validation with predetermined train-test splits,
        which is useful for consistent benchmarking and when specific data
        partitioning is required.

        Parameters
        ----------
        x_train : np.ndarray
            Training input features of shape (n_train_samples, n_input_features).
        y_train : np.ndarray
            Training target values of shape (n_train_samples, n_output_features).
        x_test : np.ndarray
            Testing input features of shape (n_test_samples, n_input_features).
        y_test : np.ndarray
            Testing target values of shape (n_test_samples, n_output_features).
        norm : {'Frobenius', 'inf'}, optional
            Type of norm to use for error calculation. Default is 'Frobenius'.
        separate_err : bool, optional
            If True, return the error for each test sample separately.
            If False, return the overall aggregated error. Default is False.
        lift_y : np.ndarray, optional
            If provided, this array is added to both true and predicted values
            before error calculation. Default is None.

        Returns
        -------
        float or np.ndarray
            The calculated norm of the prediction error. If separate_err=True,
            returns an array of errors for each test sample.

        Raises
        ------
        AssertionError
            If an invalid norm type is specified.
        """
        self.fit(x_train, y_train)

        if norm == "Frobenius":
            return self.frobenius_norm(
                x_test, y_test, separate_err=separate_err, lift_y=lift_y
            )
        elif norm == "inf":
            return self.inf_norm(x_test, y_test, separate_err=separate_err)
        else:
            print("Please enter variable norm with value 'Frobenius' or 'inf'")
            assert False

    def multi_validate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ranks: List[int],
        training_ratio: float = 0.8,
        rand_seed: int = 42,
        norm: str = "Frobenius",
        separate_err: bool = False,
        lift_y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Validate the model performance across multiple POD ranks.

        This method systematically evaluates model performance for different
        numbers of POD modes, which is useful for determining the optimal
        rank-accuracy trade-off.

        Parameters
        ----------
        x : np.ndarray
            Input features of shape (n_samples, n_input_features).
        y : np.ndarray
            Target values of shape (n_samples, n_output_features).
        ranks : list of int
            List of POD ranks to evaluate.
        training_ratio : float, optional
            Fraction of data to use for training. Default is 0.8.
        rand_seed : int, optional
            Random seed for reproducible train-test splits. Default is 42.
        norm : {'Frobenius', 'inf'}, optional
            Type of norm to use for error calculation. Default is 'Frobenius'.
        separate_err : bool, optional
            If True, return the error for each test sample separately for each rank.
            If False, return overall aggregated errors. Default is False.
        lift_y : np.ndarray, optional
            If provided, this array is added to both true and predicted values
            before error calculation. Default is None.

        Returns
        -------
        np.ndarray
            Array of validation errors corresponding to each rank in the input list.

        Raises
        ------
        AssertionError
            If an invalid norm type is specified.

        Notes
        -----
        The method uses the same train-test split for all ranks to ensure
        fair comparison. The original rank setting is modified during the
        process and should be reset if needed after calling this method.

        Examples
        --------
        >>> ranks = [5, 10, 15, 20, 25]
        >>> errors = model.multi_validate(X, Y, ranks, training_ratio=0.75)
        >>> optimal_rank = ranks[np.argmin(errors)]
        >>> print(f"Optimal rank: {optimal_rank}")
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=training_ratio, random_state=rand_seed
        )
        errors = []
        for i in ranks:
            self.rank = i
            self.fit(x_train, y_train)
            if norm == "Frobenius":
                errors.append(
                    self.frobenius_norm(
                        x_test, y_test, separate_err=separate_err, lift_y=lift_y
                    )
                )
            elif norm == "inf":
                errors.append(self.inf_norm(x_test, y_test, separate_err=separate_err))
            else:
                print("Please enter variable norm with value 'Frobenius' or 'inf'")
                assert False
        return np.array(errors)

    def multi_validate_fixed(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        ranks: List[int],
        norm: str = "Frobenius",
        separate_err: bool = False,
        lift_y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Validate the model across multiple POD ranks with fixed datasets.

        This method combines the functionality of multi-rank validation with
        fixed train-test splits, providing consistent evaluation across different
        POD ranks using predetermined data partitions.

        Parameters
        ----------
        x_train : np.ndarray
            Training input features of shape (n_train_samples, n_input_features).
        y_train : np.ndarray
            Training target values of shape (n_train_samples, n_output_features).
        x_test : np.ndarray
            Testing input features of shape (n_test_samples, n_input_features).
        y_test : np.ndarray
            Testing target values of shape (n_test_samples, n_output_features).
        ranks : list of int
            List of POD ranks to evaluate.
        norm : {'Frobenius', 'inf'}, optional
            Type of norm to use for error calculation. Default is 'Frobenius'.
        separate_err : bool, optional
            If True, return the error for each test sample separately for each rank.
            If False, return overall aggregated errors. Default is False.
        lift_y : np.ndarray, optional
            If provided, this array is added to both true and predicted values
            before error calculation. Default is None.

        Returns
        -------
        np.ndarray
            Array of validation errors corresponding to each rank. Shape depends
            on separate_err: if False, shape is (len(ranks),); if True, shape is
            (len(ranks), n_test_samples).

        Raises
        ------
        AssertionError
            If an invalid norm type is specified.

        Notes
        -----
        This method is particularly useful for:
        - Systematic rank selection studies
        - Benchmarking with consistent datasets
        - Error analysis across different dimensionality reductions

        The original rank setting is modified during the process and should be
        reset if needed after calling this method.
        """
        errors = []
        for i in ranks:
            self.rank = i
            self.fit(x_train, y_train)
            if norm == "Frobenius":
                errors.append(
                    self.frobenius_norm(
                        x_test, y_test, separate_err=separate_err, lift_y=lift_y
                    )
                )
            elif norm == "inf":
                errors.append(self.inf_norm(x_test, y_test, separate_err=separate_err))
            else:
                print("Please enter variable norm with value 'Frobenius' or 'inf'")
                assert False
        return np.array(errors)

    def check_input(self, x: np.ndarray) -> np.ndarray:
        tolerance = 0.5
        list_warning = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i, j] > 1 + tolerance:
                    list_warning.append(np.array([i, j, x[i, j]]))
                    x[i, j] = 1 + tolerance
                elif x[i, j] < -tolerance:
                    list_warning.append(np.array([i, j, x[i, j]]))
                    x[i, j] = -tolerance

        # if len(list_warning) > 0:
        #     warnings.warn(
        #         f"Some input features are out of the expected range [0, 1]. "
        #         f"Values have been clipped to [-{tolerance}, {1 + tolerance}]. "
        #         f"Details (sample index, feature index, original value): {list_warning}"
        #     )

        self.list_warning = list_warning
        return x
