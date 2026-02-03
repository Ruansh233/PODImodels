# PODImodels

[![PyPI version](https://badge.fury.io/py/PODImodels.svg)](https://pypi.org/project/PODImodels/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PODImodels** (Proper Orthogonal Decomposition-based Interpolation Models) is a Python package for creating Reduced-Order Models (ROM) using POD combined with various machine learning techniques. It provides a unified framework for reduced-order modeling of high-dimensional field data, particularly useful for computational fluid dynamics and other physics-based simulations.

## Features

- **Multiple Interpolation Methods**: Support for Linear Regression, Ridge Regression, Gaussian Process Regression (GPR), Radial Basis Functions (RBF), and Artificial Neural Networks (ANN)
- **Flexible Modeling Approaches**: Choose between direct field prediction or POD coefficient interpolation
- **Easy-to-Use API**: Scikit-learn-inspired interface with `fit()` and `predict()` methods
- **Built-in Scaling**: Optional data normalization for improved model performance
- **Extensible Architecture**: Abstract base class for implementing custom interpolation models

## Installation

Install PODImodels using pip:

```bash
pip install PODImodels
```

For development installation:

```bash
git clone https://github.com/Ruansh233/PODImodels.git
cd PODImodels
pip install -e .
```

## Requirements

- Python >= 3.8
- numpy
- scikit-learn
- scipy
- pyvista
- torch

All dependencies will be automatically installed with the package.

## Quick Start

```python
import numpy as np
from PODImodels import PODGPR, fieldsRBF

# Prepare your data
# parameters: (n_samples, n_parameters)
# field_snapshots: (n_samples, n_field_points)
parameters = np.random.rand(100, 3)
field_snapshots = np.random.rand(100, 10000)

# Example 1: POD with Gaussian Process Regression
model = PODGPR(rank=15, with_scaler_x=True, with_scaler_y=True)
model.fit(parameters, field_snapshots)
predictions = model.predict(new_parameters)

# Example 2: Direct field prediction with Radial Basis Functions
model = fieldsRBF(kernel='thin_plate_spline', degree=2)
model.fit(parameters, field_snapshots)
field_prediction = model.predict(test_parameters)
```

## Available Models

PODImodels provides two categories of models:

1. **Direct Field Models** (`fields*`): Learn direct mapping from parameters to field values
2. **POD Coefficient Models** (`POD*`): Learn mapping from parameters to POD coefficients (more efficient for large fields)

### Linear Regression Models
- `fieldsLinear`: Direct field prediction using linear regression
- `PODLinear`: POD coefficient prediction using linear regression
- `fieldsRidge`: Direct field prediction using Ridge regression
- `PODRidge`: POD coefficient prediction using Ridge regression

### Gaussian Process Regression Models
- `fieldsGPR`: Direct field prediction using Gaussian Process Regression
- `PODGPR`: POD coefficient prediction using Gaussian Process Regression
- `fieldsRidgeGPR`: Field prediction with Ridge regularization and GPR
- `PODRidgeGPR`: POD coefficient prediction with Ridge regularization and GPR

### Radial Basis Function Models
- `fieldsRBF`: Direct field prediction using Radial Basis Function interpolation
- `PODRBF`: POD coefficient prediction using Radial Basis Function interpolation
- `fieldsRidgeRBF`: Field prediction with Ridge regularization and RBF
- `PODRidgeRBF`: POD coefficient prediction with Ridge regularization and RBF

### Neural Network Models
- `PODANN`: POD coefficient prediction using Artificial Neural Networks

## Usage Examples

### Example 1: POD-GPR for CFD Field Reconstruction

```python
from PODImodels import PODGPR
import numpy as np

# Load your simulation data
# X: Design parameters (e.g., Reynolds number, geometry params)
# Y: Field snapshots (e.g., velocity, pressure fields)
X_train = np.load('parameters.npy')  # shape: (n_samples, n_params)
Y_train = np.load('fields.npy')      # shape: (n_samples, n_points)

# Initialize model with 20 POD modes
model = PODGPR(rank=20, with_scaler_x=True, with_scaler_y=True)

# Train the model
model.fit(X_train, Y_train)

# Predict new fields
X_test = np.array([[100, 0.5, 1.2]])  # New parameter set
Y_pred = model.predict(X_test)
```

### Example 2: Direct Field Interpolation with RBF

```python
from PODImodels import fieldsRBF

# Initialize RBF model
model = fieldsRBF(
    kernel='thin_plate_spline',
    degree=2,
    with_scaler_x=True
)

# Train and predict
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
```

### Example 3: Using Neural Networks

```python
from PODImodels import PODANN

# Initialize ANN model
model = PODANN(
    rank=15,
    hidden_layers=[64, 32],
    activation='relu',
    epochs=100
)

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
```

## Model Selection Guide

- **Linear/Ridge**: Fast, interpretable, works well for linear relationships
- **GPR**: Best for smooth, continuous fields with uncertainty quantification
- **RBF**: Excellent for scattered data interpolation
- **ANN**: Powerful for complex nonlinear relationships, requires more data

Choose **POD*** models when dealing with large field dimensions (>1000 points) for better computational efficiency.

## Documentation

For detailed documentation and API reference, visit the [GitHub repository](https://github.com/Ruansh233/PODImodels).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use PODImodels in your research, please cite:

```bibtex
@software{PODImodels2024,
  author = {Ruan, Shenhui},
  title = {PODImodels: POD-based Interpolation Models for Reduced-Order Modeling},
  year = {2024},
  url = {https://github.com/Ruansh233/PODImodels}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Shenhui Ruan**  
Email: shenhui.ruan@kit.edu  
GitHub: [@Ruansh233](https://github.com/Ruansh233)

## Acknowledgments

This package was developed for reduced-order modeling in computational physics applications, with a focus on CFD simulations.
