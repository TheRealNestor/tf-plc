"""
Remove DequantizeLinear layers that only operate on constant weights.
"""

import logging
from ..base_pass import OptimizationPass
from ...types import NetworkIR, DequantizeLinearLayer

logger = logging.getLogger(__name__)


class RemoveWeightDequantPass(OptimizationPass):
    """
    Remove DequantizeLinear layers that operate on constants (weights/biases).

    These are redundant because we can dequantize weights at code generation time
    rather than at runtime.
    """

    def get_name(self) -> str:
        return "remove_weight_dequant"

    def optimize(self, ir: NetworkIR) -> None:
        """Find and mark weight dequantization layers for removal."""
        for name, layer in ir.layers.items():
            if isinstance(layer, DequantizeLinearLayer):
                # Check if input is a constant (no producer layer)
                if len(layer.inputs) > 0:
                    input_tensor = layer.inputs[0]
                    producer = ir.tensor_producers.get(input_tensor)

                    # If no producer, it's a constant/initializer
                    if producer is None:
                        self.bypass_layer(layer)
                        logger.debug(
                            f"Marked weight dequantization for removal: {name}"
                        )

        if self.removed_layers:
            logger.info(
                f"Found {len(self.removed_layers)} weight dequantization layers to remove"
            )
