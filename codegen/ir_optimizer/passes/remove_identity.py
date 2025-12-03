"""
Remove Identity layers (pass-through operations).
"""

import logging
from ..base_pass import OptimizationPass
from ...types import NetworkIR

logger = logging.getLogger(__name__)


class RemoveIdentityPass(OptimizationPass):
    """Remove Identity layers that just pass data through unchanged."""

    def get_name(self) -> str:
        return "remove_identity"

    def run(self, ir: NetworkIR) -> None:
        """Find and mark all Identity layers for removal."""
        for name, layer in ir.layers.items():
            if layer.op_type == "Identity":
                self.mark_for_removal(layer)
                logger.debug(f"Marked Identity layer for removal: {name}")

        if self.removed_layers:
            logger.info(f"Found {len(self.removed_layers)} Identity layers to remove")
