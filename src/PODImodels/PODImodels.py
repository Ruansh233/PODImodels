"""
Compatibility module for legacy imports.

This module keeps the old import path `PODImodels.PODImodels` stable while
the concrete implementations now live under `PODImodels.models`.
"""

from .models import (
    PODANN,
    PODGPR,
    PODLinear,
    PODRBF,
    PODRidge,
    PODRidgeGPR,
    PODRidgeRBF,
    fieldsGPR,
    fieldsLinear,
    fieldsRBF,
    fieldsRidge,
    fieldsRidgeGPR,
    fieldsRidgeRBF,
)

__all__ = [
    "fieldsLinear",
    "PODLinear",
    "fieldsRidge",
    "PODRidge",
    "fieldsGPR",
    "PODGPR",
    "fieldsRidgeGPR",
    "PODRidgeGPR",
    "fieldsRBF",
    "PODRBF",
    "fieldsRidgeRBF",
    "PODRidgeRBF",
    "PODANN",
]
