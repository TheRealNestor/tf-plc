"""
Remove redundant QuantizeLinear → DequantizeLinear pairs.

These pairs cancel out (quantize then immediately dequantize) and can be eliminated.
"""

import numpy as np
from ..base_pass import OptimizationPass
from ...types import NetworkIR, QuantizeLinearLayer, DequantizeLinearLayer

import logging

logger = logging.getLogger(__name__)


class RemoveRedundantQuantPairPass(OptimizationPass):
    """Remove redundant QuantizeLinear → DequantizeLinear pairs that cancel out."""

    def get_name(self) -> str:
        return "remove_redundant_quant_pair"

    def optimize(self, network: NetworkIR) -> None:
        """Find and mark redundant Quant-Dequant pairs for removal."""

        pairs_found = 0

        for layer_name in network.execution_order:
            layer = network.get_layer(layer_name)

            if isinstance(layer, QuantizeLinearLayer):
                # Check if next layer is Dequantize with same params
                consumers = network.get_output_layers(layer_name)

                if len(consumers) == 1:
                    next_layer = network.get_layer(consumers[0])

                    if isinstance(next_layer, DequantizeLinearLayer):
                        # Compare quantization parameters (handle shape differences)
                        scale_match = np.allclose(
                            np.atleast_1d(layer.scale),
                            np.atleast_1d(next_layer.scale),
                            rtol=1e-9,
                        )
                        zp_match = np.array_equal(
                            np.atleast_1d(layer.zero_point),
                            np.atleast_1d(next_layer.zero_point),
                        )

                        if scale_match and zp_match:
                            self.mark_for_removal(layer)
                            self.mark_for_removal(next_layer)
                            pairs_found += 1
                            logger.debug(
                                f"Removing redundant Quant-Dequant pair: {layer_name} -> {consumers[0]}"
                            )

        if pairs_found > 0:
            logger.info(f"Found {pairs_found} redundant Quant-Dequant pairs to remove")
