"""
Fuse QuantizeLinear → DequantizeLinear pairs.
"""

import numpy as np
from ..base_pass import OptimizationPass
from ...types import NetworkIR, QuantizeLinearLayer, DequantizeLinearLayer

import logging
logger = logging.getLogger(__name__)

class FuseQuantDequantPass(OptimizationPass):
    """Remove redundant QuantizeLinear → DequantizeLinear pairs."""

    def get_name(self) -> str:
        return "fuse_quant_dequant"

    def optimize(self, network: NetworkIR) -> None:
        """Find and mark Quant-Dequant pairs for removal."""

        for layer_name in network.execution_order:
            layer = network.get_layer(layer_name)

            if isinstance(layer, QuantizeLinearLayer):
                # check if next layer is Dequantize with same params
                consumers = network.get_output_layers(layer_name)

                if len(consumers) == 1:
                    next_layer = network.get_layer(consumers[0])
                
                    if isinstance(next_layer, DequantizeLinearLayer):
                        if (np.array_equal(layer.scale, next_layer.scale) and
                            np.array_equal(layer.zero_point, next_layer.zero_point)):
                            self.mark_for_removal(layer)
                            self.mark_for_removal(next_layer)
                            logger.debug(
                                f"Fusing Quant-Dequant pair and removing: {layer_name} -> {consumers[0]}"
                            )

