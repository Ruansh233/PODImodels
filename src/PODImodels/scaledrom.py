"""
Scaled Reduced Order Models
===========================

This module provides utilities for applying scaling transformations to reduced-order
models, enabling better numerical conditioning and improved model performance through
data preprocessing.

Classes
-------
maxMinScalar
    A simple min-max scaler for data normalization.
scaledROM
    A wrapper class that applies scaling to reduced-order models.

Notes
-----
The scaling approach can improve model performance by:
- Normalizing data to a consistent range [0, 1]
- Improving numerical stability of machine learning algorithms
- Ensuring all features contribute equally to the learning process
"""

import numpy as np
from sklearn.model_selection import train_test_split


class maxMinScalar:
    """
    A simple min-max scaler for data normalization.

    This class provides min-max scaling functionality to transform data to the
    range [0, 1]. It computes and stores the minimum and maximum values during
    initialization and provides methods for forward and inverse transformations.

    Parameters
    ----------
    data : np.ndarray
        Input data array for which min-max scaling parameters are computed.

    Attributes
    ----------
    data : np.ndarray
        The original input data.
    min : float
        The minimum value in the data.
    max : float
        The maximum value in the data.

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> scaler = maxMinScalar(data)
    >>> scaled_data = scaler.fit_transform()
    >>> print(scaled_data)  # [0.0, 0.25, 0.5, 0.75, 1.0]
    >>> original_data = scaler.inverse_transform(scaled_data)
    """

    def __init__(self, data):
        self.data = data
        self.min = np.min(data)
        self.max = np.max(data)

    def fit_transform(self):
        """
        Transform the stored data using min-max scaling.

        Returns
        -------
        np.ndarray
            Scaled data in the range [0, 1].
        """
        return (self.data - self.min) / (self.max - self.min)

    def transform(self, x):
        """
        Transform new data using the stored scaling parameters.

        Parameters
        ----------
        x : np.ndarray
            Data to be transformed using the same scaling parameters.

        Returns
        -------
        np.ndarray
            Transformed data in the range [0, 1].
        """
        return (x - self.min) / (self.max - self.min)

    def inverse_transform(self, x):
        """
        Inverse transform scaled data back to original scale.

        Parameters
        ----------
        x : np.ndarray
            Scaled data in the range [0, 1].

        Returns
        -------
        np.ndarray
            Data transformed back to the original scale.
        """
        return x * (self.max - self.min) + self.min


class scaledROM:
    """
    A wrapper class for applying scaling transformations to reduced-order models.

    This class provides a standardized interface for combining data scaling with
    any reduced-order model (ROM). It handles the scaling of training data, applies
    the underlying ROM, and provides validation methods with proper inverse scaling
    for error computation.

    Parameters
    ----------
    scalar : class
        A scaler class (like maxMinScalar) that provides fit_transform, transform,
        and inverse_transform methods.
    ROM : object
        A reduced-order model object that implements fit, predict, and validate methods.

    Attributes
    ----------
    scalar : class
        The scaler class for data transformation.
    ROM : object
        The underlying reduced-order model.
    tmpscalar : object
        Instance of the scaler fitted to the training data.

    Examples
    --------
    >>> from PODImodels import PODGPR
    >>> base_model = PODGPR(rank=10)
    >>> scaled_model = scaledROM(maxMinScalar, base_model)
    >>> scaled_model.fit(parameters, field_data)
    >>> predictions = scaled_model.predict(new_parameters)
    
    >>> # Validation with proper error scaling
    >>> error = scaled_model.validate(parameters, field_data, training_ratio=0.8)
    """

    def __init__(self, scalar, ROM):
        self.scalar = scalar
        self.ROM = ROM

    def scale_data(self, y):
        """
        Apply scaling to the target data.

        Parameters
        ----------
        y : np.ndarray
            Target data to be scaled.

        Returns
        -------
        np.ndarray
            Scaled target data.
        """
        self.tmpscalar = self.scalar(y)
        return self.tmpscalar.fit_transform()

    def data_split(self, x, y, train_size):
        """
        Split data into training and testing sets.

        Parameters
        ----------
        x : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        train_size : float
            Fraction of data to use for training.

        Returns
        -------
        tuple
            Tuple containing (x_train, x_test, y_train, y_test).
        """
        return train_test_split(x, y, train_size=train_size, random_state=42)

    def fit(self, x, y):
        """
        Fit the scaled reduced-order model.

        This method applies scaling to the target data and then fits the
        underlying ROM to the scaled data.

        Parameters
        ----------
        x : np.ndarray
            Input features of shape (n_samples, n_input_features).
        y : np.ndarray
            Target values of shape (n_samples, n_output_features).
        """
        data = self.scale_data(y)
        self.ROM.fit(x, data)

    def predict(self, x):
        """
        Make predictions using the scaled ROM.

        Parameters
        ----------
        x : np.ndarray
            Input features for prediction.

        Returns
        -------
        np.ndarray
            Predicted values in the scaled space. Use inverse_transform
            to convert back to original scale if needed.

        Notes
        -----
        The predictions are returned in the scaled space. For predictions
        in the original scale, apply self.tmpscalar.inverse_transform()
        to the results.
        """
        return self.ROM.predict(x)

    def validate(self, x, y, training_ratio=0.8, rand_seed=42, norm="Frobenius"):
        """
        Validate the scaled ROM with automatic train-test splitting.

        This method handles the complete validation pipeline including data scaling,
        train-test splitting, model fitting, prediction, and error computation in
        the original (unscaled) space.

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

        Returns
        -------
        float
            Validation error computed in the original data space.

        Raises
        ------
        AssertionError
            If an invalid norm type is specified.

        Notes
        -----
        The error is computed in the original (unscaled) space by applying
        inverse transformations to both true and predicted values. This
        ensures that the validation error is meaningful and comparable
        across different scaling approaches.
        """
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
