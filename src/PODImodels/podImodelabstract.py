"""
Compatibility module for legacy imports.

This module keeps the old import path `PODImodels.podImodelabstract` stable while
the implementation now lives in `PODImodels.base`.
"""

from .base import PODImodelAbstract

__all__ = ["PODImodelAbstract"]
