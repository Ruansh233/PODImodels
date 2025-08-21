import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from typing import Optional
from sklearn.gaussian_process.kernels import Kernel
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from scipy.linalg import svd
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from podImodelabstract import PODImodelAbstract
import torch
import torch.nn as nn
import torch.optim as optim
import random


class fieldsLinear(PODImodelAbstract):
    """A simple linear regression model for fields."""

    def __init__(self):
        self.lin = LinearRegression()

    def fit(self, x, y):
        self.lin.fit(x, y)

    def predict(self, x):
        return self.lin.predict(x)


class PODLinear(PODImodelAbstract):
    """A linear regression model for POD coefficients."""

    def __init__(self, rank: int = 10, with_scalar: bool = False):
        """
        Initialize the PODLinear model.

        Args:
            rank (int): The rank for POD.
            with_scalar (bool): Whether to include a scalar in the model.
        """
        self.lin = LinearRegression()
        self.rank = rank
        self.with_scalar = with_scalar

    def fit(self, x, y):
        y = self.performPOD(y)

        if self.with_scalar:
            self.coeffs_scalar = MinMaxScaler()
            y = self.coeffs_scalar.fit_transform(y)

        self.lin.fit(x, y)

    def predict(self, x):
        if self.with_scalar:
            return self.coeffs_scalar.inverse_transform(self.lin.predict(x)) @ self.v
        else:
            return self.lin.predict(x) @ self.v


class fieldsRidge(PODImodelAbstract):
    """A Ridge regression model for fields."""
    def __init__(self):
        self.lin = Ridge()

    def fit(self, x, y):
        self.lin.fit(x, y)

    def predict(self, x):
        return self.lin.predict(x)


class PODRidge(PODImodelAbstract):
    """A Ridge regression model for POD coefficients."""
    def __init__(self, rank: int = 10, with_scalar: bool = False):
        """
        Initialize the PODRidge model.

        Args:
            rank (int): The rank for POD.
            with_scalar (bool): Whether to include a scalar in the model.
        """
        self.lin = Ridge()
        self.rank = rank
        self.with_scalar = with_scalar

    def fit(self, x, y):
        y = self.performPOD(y)

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
    """A Gaussian Process Regression model for fields."""
    def __init__(self, kernel: Optional[Kernel] = None, alpha: float = 1.0e-10):
        """
        Initialize the fieldsGPR model.

        Args:
            kernel (Optional[Kernel]): The kernel to use for the GPR.
            alpha (float): The noise level for the GPR.
        """
        if kernel is None:
            self.kernel = RBF(length_scale=1.0e0, length_scale_bounds="fixed")
        else:
            self.kernel = kernel

        self.alpha = alpha

    def fit(self, x, y):
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
        self.gpr.fit(x, y)

    def predict(self, x):
        return self.gpr.predict(x)


class PODGPR(PODImodelAbstract):
    def __init__(
        self,
        kernel: Optional[Kernel] = None,
        alpha: float = 1.0e-10,
        rank: int = 10,
        with_scalar: bool = False,
    ):
        if kernel is None:
            self.kernel = RBF(length_scale=1.0e0, length_scale_bounds="fixed")
        else:
            self.kernel = kernel
        self.alpha = alpha
        self.rank = rank
        self.with_scalar = with_scalar

    def fit(self, x, y):
        y = self.performPOD(y)
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)

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
    def __init__(self, kernel: Optional[Kernel] = None, alpha: float = 1.0e-10):
        if kernel is None:
            self.kernel = RBF(length_scale=1.0e0, length_scale_bounds="fixed")
        else:
            self.kernel = kernel

        self.alpha = alpha

    def fit(self, x, y):
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
        self.lin = Ridge()

        self.lin.fit(x, y)
        self.gpr.fit(x, y - self.lin.predict(x))

    def predict(self, x):
        return self.gpr.predict(x) + self.lin.predict(x)


class PODRidgeGPR(PODImodelAbstract):
    def __init__(
        self,
        kernel: Optional[Kernel] = None,
        alpha: float = 1.0e-10,
        rank: int = 10,
        with_scalar: bool = False,
    ):
        if kernel is None:
            self.kernel = RBF(length_scale=1.0e0, length_scale_bounds="fixed")
        else:
            self.kernel = kernel
        self.alpha = alpha
        self.rank = rank
        self.with_scalar = with_scalar

    def fit(self, x, y):
        y = self.performPOD(y)
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha)
        self.lin = Ridge()

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
    def __init__(self, kernel: str = "linear", epsilon: float = 1.0):
        self.kernel = kernel
        self.epsilon = epsilon

    def fit(self, x, y):
        self.rbf = RBFInterpolator(x, y, kernel=self.kernel, epsilon=self.epsilon)

    def predict(self, x):
        return self.rbf(x)


class PODRBF(PODImodelAbstract):
    def __init__(
        self,
        kernel: str = "linear",
        epsilon: float = 1.0,
        rank: int = 10,
        with_scalar: bool = False,
        neighbors: Optional[int] = None,
    ):
        self.kernel = kernel
        self.epsilon = epsilon
        self.neighbors = neighbors
        self.rank = rank
        self.with_scalar = with_scalar

    def fit(self, x, y):
        y = self.performPOD(y)

        if self.with_scalar:
            self.coeffs_scalar = MinMaxScaler()
            y = self.coeffs_scalar.fit_transform(y)

        self.rbf = RBFInterpolator(
            x, y, kernel=self.kernel, epsilon=self.epsilon, neighbors=self.neighbors
        )

    def predict(self, x):
        if self.with_scalar:
            tmp = self.coeffs_scalar.inverse_transform(self.rbf(x))
            return tmp @ self.v
        else:
            return self.rbf(x) @ self.v


class fieldsRidgeRBF(PODImodelAbstract):
    def __init__(self, kernel: str = "linear", epsilon: float = 1.0):
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
    def __init__(
        self,
        kernel: str = "linear",
        epsilon: float = 1.0,
        rank: int = 10,
        with_scalar: bool = False,
    ):
        self.kernel = kernel
        self.epsilon = epsilon
        self.lin = Ridge()
        self.rank = rank
        self.with_scalar = with_scalar

    def fit(self, x, y):
        y = self.performPOD(y)

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


class PODANN(PODImodelAbstract):
    """
    A PyTorch-based Artificial Neural Network for interpolating POD coefficients.

    This class builds, trains, and uses a fully connected neural network
    to map input parameters (e.g., CFD boundary conditions) to POD coefficients.
    It handles data normalization, various activation functions, loss functions,
    and optimizers.
    """

    def __init__(
        self,
        rank: int = 10,
        hidden_layer_sizes: list = None,
        activation_function_name: str = "relu",
        learning_rate: float = 0.001,
        loss_function_name: str = "mse",
        optimizer_name: str = "adam",
        num_epochs: int = 1000,
        stop_threshold: float = 1e-4,
        random_seed: int = 42,
    ):
        """
        Initializes the PODANN.

        Args:
            rank (int): The number of POD modes to use for the model.
                        Defaults to 10.
            hidden_layer_sizes (list): A list where each element is the number
                                       of neurons in a corresponding hidden layer.
                                       Example: [32, 16] for two hidden layers
                                       with 32 and 16 neurons respectively.
                                       Defaults to [32, 16] if None.
            activation_function_name (str): Name of the activation function to use
                                            in hidden layers ('relu', 'sigmoid', 'tanh').
                                            Defaults to 'relu'.
            learning_rate (float): The learning rate for the optimizer. Defaults to 0.001.
            loss_function_name (str): Name of the loss function ('mse' for Mean Squared Error,
                                      'l1' for Mean Absolute Error). Defaults to 'mse'.
            optimizer_name (str): Name of the optimizer ('adam', 'sgd'). Defaults to 'adam'.
            num_epochs (int): The number of training epochs. Defaults to 1000.
            stop_threshold (float): Threshold for early stopping based on loss value.
                                   Defaults to 1e-6.
            random_seed (int): Random seed for reproducibility. Defaults to 42.
        """
        self.rank = rank
        self.hidden_layer_sizes = (
            hidden_layer_sizes if hidden_layer_sizes is not None else [32, 16]
        )
        self.activation_function_name = activation_function_name.lower()
        self.learning_rate = learning_rate
        self.loss_function_name = loss_function_name.lower()
        self.optimizer_name = optimizer_name.lower()
        self.num_epochs = num_epochs
        self.stop_threshold = stop_threshold
        self.random_seed = random_seed

        # Determine the device to use (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # --- Added: Print current PyTorch CPU thread count ---
        print(
            f"PyTorch is configured to use {torch.get_num_threads()} CPU threads for this process."
        )
        # ---------------------------------------------------

        # Initialize model and scalers to None, they will be set during fit
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Set random seed for reproducibility
        self._set_all_seeds(self.random_seed)

    def _get_activation_function(self):
        """Returns the PyTorch activation function module based on its name."""
        if self.activation_function_name == "relu":
            return nn.ReLU()
        elif self.activation_function_name == "sigmoid":
            return nn.Sigmoid()
        elif self.activation_function_name == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(
                f"Unsupported activation function: {self.activation_function_name}"
            )

    def _get_loss_function(self):
        """Returns the PyTorch loss function module based on its name."""
        if self.loss_function_name == "mse":
            return nn.MSELoss()
        elif self.loss_function_name == "l1":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function_name}")

    def _get_optimizer(self, model_parameters):
        """Returns the PyTorch optimizer based on its name and model parameters."""
        if self.optimizer_name == "adam":
            return optim.Adam(model_parameters, lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            return optim.SGD(model_parameters, lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _build_model(self, input_dim: int, output_dim: int):
        """
        Builds the neural network model dynamically based on specified hidden layers.

        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output features (POD coefficients).
        """
        layers = []
        current_dim = input_dim
        activation = self._get_activation_function()

        for h_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(current_dim, h_size))
            layers.append(activation)
            current_dim = h_size

        # Output layer with linear activation (regression problem)
        layers.append(nn.Linear(current_dim, output_dim))

        self.model = nn.Sequential(*layers).to(self.device)
        print("\n--- Model Architecture ---")
        print(self.model)
        print("--------------------------\n")

    def _set_all_seeds(self, seed: int):
        """
        Sets the random seed for reproducibility across different libraries.
        Args:
            seed (int): The seed value to use.
        """
        np.random.seed(seed)  # NumPy seed
        random.seed(seed)  # Python's built-in random module seed
        torch.manual_seed(seed)  # PyTorch CPU seed
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # PyTorch GPU seed
            torch.cuda.manual_seed_all(seed)  # PyTorch Multi-GPU seed
            # Optional: For deterministic CUDA operations, but can slow down training
            # If you encounter issues, you might need to comment these out.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"CUDA deterministic set to {torch.backends.cudnn.deterministic}")

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Trains the neural network model.

        Args:
            X_train (np.ndarray): Training input data (parameters).
                                  Shape: (num_samples, input_dim).
            y_train (np.ndarray): Training output data (high fidelity data).
                                  Shape: (num_samples, output_dim).
        """

        # perform POD reduction if not already done
        y = self.performPOD(y)

        if x.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X_train and y_train must match.")
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("Input and output data must be 2D numpy arrays.")

        input_dim = x.shape[1]
        output_dim = y.shape[1]

        # 1. Normalize input and output data
        X_train_scaled = self.scaler_X.fit_transform(x)
        y_train_scaled = self.scaler_y.fit_transform(y)

        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(
            self.device
        )
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(
            self.device
        )

        # 2. Build the model
        self._build_model(input_dim, output_dim)

        # 3. Define loss function and optimizer
        criterion = self._get_loss_function()
        optimizer = self._get_optimizer(self.model.parameters())

        # 4. Training loop
        print("Starting model training...")
        for epoch in range(self.num_epochs):
            # Set model to training mode
            self.model.train()

            # Forward pass
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)

            # Backward and optimize
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            if (epoch + 1) % 1000 == 0 or epoch == 0:
                # Relative L2 norm loss (optional)
                relative_loss = torch.sqrt(loss) / torch.norm(y_train_tensor)
                print(
                    f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.6f}, Relative L2 norm loss: {relative_loss.item():.6f}"
                )

                # Early stopping condition (optional)
                if (
                    relative_loss < self.stop_threshold
                ):  # Arbitrary threshold for early stopping
                    print("Early stopping triggered.")
                    break

        print("Model training finished.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts POD coefficients for new input data using the trained model.

        Args:
            x (np.ndarray): Test input data (parameters).
                                 Shape: (num_samples, input_dim).

        Returns:
            np.ndarray: Predicted POD coefficients.
                        Shape: (num_samples, output_dim).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call .fit() first.")
        if x.ndim != 2:
            raise ValueError("Test input data must be a 2D numpy array.")

        # Set model to evaluation mode (important for layers like Dropout if they were used)
        self.model.eval()

        # Normalize test input data
        X_test_scaled = self.scaler_X.transform(x)

        # Convert to PyTorch tensor
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():  # Disable gradient calculation during inference
            predictions_scaled = self.model(X_test_tensor).cpu().numpy()

        # De-normalize the predictions
        predictions = self.scaler_y.inverse_transform(predictions_scaled)

        # return high fidelity predictions
        return predictions @ self.v
