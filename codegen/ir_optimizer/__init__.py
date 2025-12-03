"""
IR optimization module.
"""

from .optimizer import IROptimizer, DEFAULT_PASSES
from .base_pass import OptimizationPass
from .passes import (
    RemoveIdentityPass,
    RemoveNoOpReshapePass,
    RemoveRedundantQuantPairPass,
    RemoveWeightDequantPass,
)

__all__ = [
    "IROptimizer",
    "DEFAULT_PASSES",
    "OptimizationPass",
    "RemoveIdentityPass",
    "RemoveNoOpReshapePass",
    "RemoveRedundantQuantPairPass",
    "RemoveWeightDequantPass",
]
