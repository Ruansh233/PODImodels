# PODImodels Workspace Instructions

## Project Overview

**PODImodels** (Proper Orthogonal Decomposition-based Interpolation Models) is a Python package for creating Reduced-Order Models (ROM) using POD combined with machine learning techniques. The framework is particularly useful for computational fluid dynamics and physics-based simulations.

### Key Features
- Multiple interpolation methods: Linear Regression, Ridge Regression, Gaussian Process Regression (GPR), Radial Basis Functions (RBF), and Artificial Neural Networks (ANN)
- Flexible modeling: direct field prediction or POD coefficient interpolation
- Scikit-learn-inspired API with `fit()` and `predict()` methods
- Built-in data scaling/normalization
- Extensible architecture with abstract base classes

## Environment & Package Management

### Using `uv` (Required)

This project uses `uv` for Python environment and dependency management. **Do NOT use `pip install` directly.**

#### Common Commands
- **Run Python**: `uv run python <script.py>`
- **Run tests**: `uv run pytest <test_file>`
- **Run modules**: `uv run python -m <module>`
- **Add dependencies**: `uv add <package>`
- **Remove dependencies**: `uv remove <package>`
- **Sync environment**: `uv sync`

#### Example Workflow
```bash
# Run the test suite
uv run pytest

# Run a specific test
uv run pytest tests/test_models.py

# Run a script with Python
uv run python examples/openfoam_cavity/benchmark_cavity_models.py

# Add a new dependency
uv add matplotlib
```

## Project Structure

```
PODImodels/
├── src/PODImodels/           # Main package source code
│   ├── base.py              # Base classes and utilities
│   ├── PODdata.py           # POD data handling
│   ├── podImodelabstract.py # Abstract model interface
│   ├── PODImodels.py        # Main model classes
│   ├── models/              # Interpolation model implementations
│   │   ├── linear.py        # Linear regression
│   │   ├── gpr.py           # Gaussian Process Regression
│   │   ├── rbf.py           # Radial Basis Functions
│   │   ├── ann.py           # Artificial Neural Networks
│   │   └── __init__.py
│   └── __init__.py
├── tests/                    # Unit and integration tests
│   ├── test_models.py
│   ├── test_imports.py
│   ├── test_openfoam_cavity_benchmark.py
│   └── conftest.py
├── examples/                 # Usage examples
│   └── openfoam_cavity/
│       ├── benchmark_cavity_models.py
│       └── README.md
└── pyproject.toml          # Project configuration
```

## Code Conventions

### POD and Mathematical Notation
- Use LaTeX notation for spatial coordinates: $\mathbf{x}$ (bold vectors using `\mathbf{}`)
- Use standard mathematical notation for reduced dimensions, modes, and coefficients
- Document POD decomposition clearly: snapshot matrix $\mathbf{X}$, modes $\mathbf{\Phi}$, coefficients $\mathbf{a}$

### API Design
- Follow scikit-learn conventions: `fit()`, `predict()`, `fit_predict()`
- Use consistent parameter naming across interpolation models
- Inherit from abstract base classes defined in `podImodelabstract.py`

### Documentation
- Docstrings use NumPy style format
- Include examples in docstrings when applicable
- Reference specific model classes and their parameters clearly

### Testing
- Write tests for new features in `tests/`
- Use `pytest` fixtures from `conftest.py` when available
- Run full test suite before committing: `uv run pytest`

## Model Selection Guide

Use the following interpolation methods based on your use case:

| Model | Use Case | Advantages | Limitations |
|-------|----------|-----------|------------|
| **Linear** | Baseline, simple relationships | Fast, interpretable | Limited expressiveness |
| **Ridge Regression** | Linear with regularization | Prevents overfitting, stable | Still assumes linearity |
| **GPR** | Non-linear with uncertainty quantification | Probabilistic, smooth | Computational cost grows with data size |
| **RBF** | Local interpolation, scattered data | Non-parametric, flexible | Sensitive to parameter tuning |
| **ANN** | Complex non-linear patterns | High expressiveness | Requires hyperparameter tuning, more training data |

## Dependencies

**Core Dependencies** (from `pyproject.toml`):
- `numpy` — numerical computation
- `scikit-learn` — machine learning utilities
- `scipy` — scientific computing
- `pyvista` — visualization (especially for CFD data)
- `torch` — neural network implementations

**Development Dependencies**:
- `pytest` — testing framework
- `foamToPython` — OpenFOAM data processing (git dependency)

## Common Tasks

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_models.py -v

# Run with coverage
uv run pytest --cov=src/PODImodels
```

### Example Workflow
```bash
# Run the OpenFOAM cavity benchmark
uv run python examples/openfoam_cavity/benchmark_cavity_models.py
```

### Debugging
- Use `uv run python -m pdb` for debugging Python scripts
- Add print statements and use logging for debugging within tests
- Check `tests/conftest.py` for test fixtures and setup

## When Creating/Modifying Code

- **New model**: Inherit from abstract base in `podImodelabstract.py` and follow existing model structure (see `models/linear.py`, `models/gpr.py`, etc.)
- **Data handling**: Use PODdata utilities from `PODdata.py` for consistent data management
- **New tests**: Place in `tests/` directory and use `uv run pytest` to verify
- **Documentation**: Update docstrings with NumPy format and include LaTeX for mathematical expressions
- **Dependencies**: Use `uv add <package>` to add new dependencies (never edit `pyproject.toml` manually)

## Git & Version Control

- Main branch is `main`
- All feature branches should have clear, descriptive names
- Run full test suite before pushing: `uv run pytest`
- Ensure code is properly tested and documented

## Performance Considerations

- POD requires SVD decomposition of snapshot matrix — can be memory-intensive for large datasets
- Consider using iterative methods for high-dimensional problems
- GPR and RBF models may have computational overhead for large datasets
- ANN models benefit from GPU acceleration via PyTorch (check CUDA availability)
