"""
PODImodels: Proper Orthogonal Decomposition based Interpolation Models
=====================================================================

A Python package for building reduced-order models using Proper Orthogonal 
Decomposition (POD) combined with various machine learning techniques for 
interpolation and prediction of high-dimensional field data.

This package provides:
- POD-based dimensionality reduction for computational fluid dynamics fields
- Various interpolation models (Gaussian Process Regression, Radial Basis Functions, Neural Networks)
- Data handling utilities for VTK/OpenFOAM field data
- Validation and visualization tools for model assessment

Main Classes
------------
PODImodelAbstract : Abstract base class
    Base class for all POD-based interpolation models.
fieldsGPR, PODGPR : Gaussian Process Regression models
    GPR models for direct field prediction and POD coefficient prediction.
fieldsRBF, PODRBF : Radial Basis Function models
    RBF models for direct field prediction and POD coefficient prediction.
PODANN : Artificial Neural Network model
    Deep learning model for POD coefficient prediction.
scaledROM : Scaled Reduced Order Model
    Wrapper for applying scaling transformations to ROM models.
PODDataSet : POD data processing
    Class for performing POD on datasets and handling modal decomposition.

Examples
--------
>>> from PODImodels import PODGPR
>>> model = PODGPR(rank=10)
>>> model.fit(parameters, field_data)
>>> predictions = model.predict(new_parameters)
"""

__all__ = [
    "PODImodelAbstract",
    "fieldsGPR",
    "PODGPR",
    "fieldsRidgeGPR",
    "PODRidgeGPR",
    "fieldsRBF",
    "PODRBF",
    "fieldsRidgeRBF",
    "PODRidgeRBF",
    "scaledROM",
    "PODANN"
]

from .podImodelabstract import PODImodelAbstract
from .PODImodels import fieldsGPR, PODGPR
from .PODImodels import fieldsRidgeGPR, PODRidgeGPR
from .PODImodels import fieldsRBF, PODRBF
from .PODImodels import fieldsRidgeRBF, PODRidgeRBF
from .PODImodels import PODANN
from .PODdata import vtk_writer, truncationErrorCal, PODDataSet, subdomainDataSet
from .scaledrom import scaledROM
