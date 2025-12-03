"""
Fuse QuantizeLinear → DequantizeLinear pairs.
"""

import logging
from ..base_pass import OptimizationPass
from ...types import NetworkIR, QuantizeLinearLayer, DequantizeLinearLayer

logger = logging.getLogger(__name__)


class FuseQuantDequantPass(OptimizationPass):
    """Remove redundant QuantizeLinear → DequantizeLinear pairs."""

    def get_name(self) -> str:
        return "fuse_quant_dequant"

    def run(self, ir: NetworkIR) -> None:
        """Find and mark Quant-Dequant pairs for removal."""
        for name, layer in ir.layers.items():
            if isinstance(layer, QuantizeLinearLayer):
                # Find consumer of this layer's output
                if len(layer.outputs) > 0:
                    output_tensor = layer.outputs[0]
                    consumers = ir.tensor_consumers.get(output_tensor, [])

                    # Only fuse if there's exactly one consumer and it's DequantizeLinear
                    if len(consumers) == 1:
                        consumer_name = consumers[0]
                        consumer = ir.layers.get(consumer_name)

                        if isinstance(consumer, DequantizeLinearLayer):
                            # Mark both for removal
                            self.mark_for_removal(layer)
                            self.mark_for_removal(consumer)
                            logger.debug(
                                f"Fusing Quant-Dequant pair: {name} -> {consumer_name}"
                            )

        if self.removed_layers:
            logger.info(
                f"Found {len(self.removed_layers) // 2} Quant-Dequant pairs to fuse"
            )
