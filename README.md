## POD-based Interpolation Models Implementation

This module contains concrete implementations of POD-based interpolation models
that inherit from PODImodelAbstract. These models combine Proper Orthogonal
Decomposition with various machine learning techniques for reduced-order modeling
of high-dimensional field data.

The models are categorized into two types:
1. Direct field models (fields*): Learn direct mapping from parameters to field values
2. POD coefficient models (POD*): Learn mapping from parameters to POD coefficients

### Available Models
#### Linear Regression Models:
- fieldsLinear: Direct field prediction using linear regression
- PODLinear: POD coefficient prediction using linear regression
- fieldsRidge: Direct field prediction using Ridge regression
- PODRidge: POD coefficient prediction using Ridge regression

#### Gaussian Process Regression Models:
- fieldsGPR: Direct field prediction using Gaussian Process Regression
- PODGPR: POD coefficient prediction using Gaussian Process Regression
- fieldsRidgeGPR: Field prediction with Ridge regularization and GPR
- PODRidgeGPR: POD coefficient prediction with Ridge regularization and GPR

#### Radial Basis Function Models:
- fieldsRBF: Direct field prediction using Radial Basis Function interpolation
- PODRBF: POD coefficient prediction using Radial Basis Function interpolation
- fieldsRidgeRBF: Field prediction with Ridge regularization and RBF
- PODRidgeRBF: POD coefficient prediction with Ridge regularization and RBF

#### Neural Network Models:
- PODANN: POD coefficient prediction using Artificial Neural Networks

### Examples
1. POD with interpolation using GPR
`
model = PODGPR(rank=15, with_scaler_x=True, with_scaler_y=True)
model.fit(parameters, field_snapshots)
predictions = model.predict(new_parameters)
`

2. Direct field prediction with RBF
`
model = fieldsRBF(kernel='thin_plate_spline', degree=2)
model.fit(parameters, field_snapshots)
field_prediction = model.predict(test_parameters)
`
