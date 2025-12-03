"""
Remove Reshape layers that don't change tensor size.
"""

import logging
import numpy as np
from ..base_pass import OptimizationPass
from ...types import NetworkIR, ReshapeLayer

logger = logging.getLogger(__name__)


class RemoveNoOpReshapePass(OptimizationPass):
    """Remove Reshape layers where input and output have same total size."""

    def get_name(self) -> str:
        return "remove_noop_reshape"

    def optimize(self, ir: NetworkIR) -> None:
        """Find and mark no-op Reshape layers for removal."""
        for name, layer in ir.layers.items():
            if isinstance(layer, ReshapeLayer):
                if layer.input_shape and layer.output_shape:
                    input_size = np.prod(layer.input_shape)
                    output_size = np.prod(layer.output_shape)

                    if input_size == output_size:
                        self.mark_for_removal(layer)
                        logger.debug(
                            f"Marked no-op Reshape for removal: {name} "
                            f"({layer.input_shape} -> {layer.output_shape})"
                        )

        if self.removed_layers:
            logger.info(
                f"Found {len(self.removed_layers)} no-op Reshape layers to remove"
            )
