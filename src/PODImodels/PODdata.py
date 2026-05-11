"""
Compatibility module for legacy imports.

This module keeps the old import path `PODImodels.PODdata` stable while
implementations live in `PODImodels.data`.
"""

from .data import PODDataSet, subdomainDataSet, truncationErrorCal, vtk_writer

__all__ = ["vtk_writer", "truncationErrorCal", "PODDataSet", "subdomainDataSet"]
