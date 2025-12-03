"""
IR optimization passes.
"""

from .remove_identity import RemoveIdentityPass
from .remove_noop_reshape import RemoveNoOpReshapePass
from .fuse_quant_dequant import FuseQuantDequantPass
from .remove_weight_dequant import RemoveWeightDequantPass

__all__ = [
    "RemoveIdentityPass",
    "RemoveNoOpReshapePass",
    "FuseQuantDequantPass",
    "RemoveWeightDequantPass",
]
