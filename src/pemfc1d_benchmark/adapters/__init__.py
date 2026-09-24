"""Model adapters: one per model this repository compares.

The interface and the standardized result object live in
:mod:`pemfc1d_benchmark.adapters.base`; each sibling module wraps one model.
"""
from pemfc1d_benchmark.adapters.base import (
    LAYERS,
    VARIABLE_LAYERS,
    VARIABLE_UNITS,
    ModelAdapter,
    ModelResult,
    Profile,
    ProfileKey,
    RunMetadata,
    get_adapter,
    registry,
)

__all__ = [
    "LAYERS", "VARIABLE_LAYERS", "VARIABLE_UNITS",
    "ModelAdapter", "ModelResult", "Profile", "ProfileKey", "RunMetadata",
    "get_adapter", "registry",
]
