"""
IR optimization passes.
"""

from .remove_identity import RemoveIdentityPass
from .remove_noop_reshape import RemoveNoOpReshapePass
from .remove_redundant_quant_pairs import RemoveRedundantQuantPairPass
from .remove_weight_dequant import RemoveWeightDequantPass
from .fuse_linear_activation import FuseLinearActivationPass

__all__ = [
    "RemoveIdentityPass",
    "RemoveNoOpReshapePass",
    "RemoveRedundantQuantPairPass",
    "RemoveWeightDequantPass",
    "FuseLinearActivationPass",
]
