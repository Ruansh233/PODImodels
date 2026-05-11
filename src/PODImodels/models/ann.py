import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..base import PODImodelAbstract


class PODANN(PODImodelAbstract):
    uses_pod: bool = True

    def __init__(
        self,
        hidden_layer_sizes: Optional[List[int]] = None,
        activation_function_name: str = "relu",
        activation_function: Optional[nn.Module] = None,
        learning_rate: float = 0.001,
        loss_function_name: str = "mse",
        optimizer_name: str = "adam",
        num_epochs: int = 1000,
        stop_threshold: float = 1e-4,
        random_seed: int = 42,
        with_weight: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_layer_sizes: Optional[List[int]] = (
            hidden_layer_sizes if hidden_layer_sizes is not None else [32, 16]
        )
        self.activation_function_name: str = activation_function_name.lower()
        self.activation_function: Optional[nn.Module] = activation_function
        self.learning_rate: float = learning_rate
        self.loss_function_name: str = loss_function_name.lower()
        self.optimizer_name: str = optimizer_name.lower()
        self.num_epochs: int = num_epochs
        self.stop_threshold: float = stop_threshold
        self.random_seed: int = random_seed
        self.with_weight: bool = with_weight

        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU Info: {torch.cuda.get_device_name(0)}")
        else:
            print(f"CPU Info: {torch.get_num_threads()} threads")

        self.model: Optional[nn.Sequential] = None
        self._set_all_seeds()

    def _get_activation_function(self) -> nn.Module:
        if (
            self.activation_function is not None
            and self.activation_function_name is not None
        ):
            raise ValueError(
                "Both activation_function and activation_function_name are set. "
                "Please use only one."
            )
        if self.activation_function is not None:
            return self.activation_function
        if self.activation_function_name == "relu":
            return nn.ReLU()
        if self.activation_function_name == "sigmoid":
            return nn.Sigmoid()
        if self.activation_function_name == "tanh":
            return nn.Tanh()
        if self.activation_function_name == "leaky_relu":
            return nn.LeakyReLU()
        if self.activation_function_name == "elu":
            return nn.ELU()
        if self.activation_function_name == "gelu":
            return nn.GELU()
        if self.activation_function_name == "softmax":
            return nn.Softmax(dim=1)
        if self.activation_function_name == "softplus":
            return nn.Softplus()
        raise ValueError(
            f"Unsupported activation function: {self.activation_function_name}"
        )

    def _get_loss_function(self) -> nn.Module:
        if self.loss_function_name == "mse":
            return nn.MSELoss(reduction="none") if self.with_weight else nn.MSELoss()
        if self.loss_function_name == "l1":
            return nn.L1Loss()
        raise ValueError(f"Unsupported loss function: {self.loss_function_name}")

    def _get_optimizer(self, model_parameters) -> optim.Optimizer:
        if self.optimizer_name == "adam":
            return optim.Adam(model_parameters, lr=self.learning_rate)
        if self.optimizer_name == "sgd":
            return optim.SGD(model_parameters, lr=self.learning_rate)
        raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _build_model(self, input_dim: int, output_dim: int) -> None:
        layers = []
        current_dim = input_dim
        activation = self._get_activation_function()
        for h_size in self.hidden_layer_sizes:
            layers.append(nn.Linear(current_dim, h_size))
            layers.append(activation)
            current_dim = h_size
        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers).to(self.device)
        print("\n--- Model Architecture ---")
        print(self.model)
        print("--------------------------\n")

    def _set_all_seeds(self) -> None:
        seed = self.random_seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"CUDA deterministic set to {torch.backends.cudnn.deterministic}")

    def fit_tmp(self, x: np.ndarray, y: np.ndarray) -> None:
        input_dim = x.shape[1]
        output_dim = y.shape[1]
        X_train_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        self._build_model(input_dim, output_dim)
        criterion = self._get_loss_function()
        optimizer = self._get_optimizer(self.model.parameters())

        if self.with_weight:
            self.coefficient_weights: np.ndarray = self.s
            if len(self.coefficient_weights) != output_dim:
                raise ValueError(
                    f"Length of coefficient_weights ({len(self.coefficient_weights)}) must match output_dim ({output_dim})."
                )
            weights_tensor: torch.Tensor = torch.tensor(
                self.coefficient_weights, dtype=torch.float32
            ).to(self.device)
            if (weights_tensor < 0).any():
                raise ValueError("Coefficient weights must be non-negative.")

        print("Starting model training...")
        for epoch in range(self.num_epochs):
            self.model.train()
            outputs = self.model(X_train_tensor)

            if self.with_weight and self.loss_function_name == "mse":
                loss_per_element = criterion(outputs, y_train_tensor)
                weighted_loss_per_element = loss_per_element * weights_tensor
                loss = weighted_loss_per_element.mean()
            elif self.with_weight:
                print(
                    "Warning: Coefficient weights are set but loss function is not MSE. Using unweighted loss."
                )
                loss = criterion(outputs, y_train_tensor)
            else:
                loss = criterion(outputs, y_train_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0 or epoch == 0:
                relative_loss = torch.sqrt(loss) / torch.norm(y_train_tensor)
                print(
                    f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.6f}, Relative L2 norm loss: {relative_loss.item():.6f}"
                )
                if relative_loss < self.stop_threshold:
                    print("Early stopping triggered.")
                    break
        print("Model training finished.")

    def predict_tmp(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call .fit() first.")

        self.model.eval()
        x = self._prepare_predict_input(x)
        x_test_tensor: torch.Tensor = torch.tensor(x, dtype=torch.float32).to(
            self.device
        )
        with torch.no_grad():
            predictions = self.model(x_test_tensor).cpu().numpy()
        return self._finalize_prediction(predictions, reconstruct_pod=True)
