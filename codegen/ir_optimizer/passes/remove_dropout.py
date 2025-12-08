"""
Remove Dropout layers - they are identity operations at inference time.
"""

import logging
from ..base_pass import OptimizationPass
from ...types import NetworkIR, DropoutLayer

logger = logging.getLogger(__name__)


class RemoveDropoutPass(OptimizationPass):
    """Remove Dropout layers since they are identity at inference."""

    def get_name(self) -> str:
        return "remove_dropout"

    def optimize(self, ir: NetworkIR) -> None:
        """Find and remove all Dropout layers."""
        for name, layer in ir.layers.items():
            if isinstance(layer, DropoutLayer):
                self.bypass_layer(layer)
                logger.debug(f"Removed Dropout layer: {name} (ratio={layer.ratio})")

        if self.removed_layers:
            logger.info(f"Removed {len(self.removed_layers)} Dropout layers")
