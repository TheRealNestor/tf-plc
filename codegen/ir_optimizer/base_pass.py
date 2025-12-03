"""
Base class for IR optimization passes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Set, Dict
from ..types import NetworkIR, BaseLayer

logger = logging.getLogger(__name__)


class OptimizationPass(ABC):
    """Base class for all optimization passes."""

    def __init__(self):
        self.removed_layers: Set[str] = set()
        self.tensor_mapping: Dict[str, str] = {}  # Maps removed tensor -> replacement

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this optimization pass."""
        pass

    @abstractmethod
    def run(self, ir: NetworkIR) -> None:
        """
        Analyze IR and mark layers for removal.

        This should populate self.removed_layers and self.tensor_mapping.
        Don't modify the IR directly - just mark what should be removed.
        """
        pass

    def mark_for_removal(self, layer: BaseLayer):
        """Mark a layer for removal and create tensor mapping."""
        self.removed_layers.add(layer.name)

        # Map output to input (pass-through)
        if len(layer.inputs) > 0 and len(layer.outputs) > 0:
            input_tensor = layer.inputs[0]
            for output_tensor in layer.outputs:
                self.tensor_mapping[output_tensor] = input_tensor
                logger.debug(f"Mapped {output_tensor} -> {input_tensor}")

    def should_remove(self, layer_name: str) -> bool:
        """Check if a layer is marked for removal."""
        return layer_name in self.removed_layers

    def get_mapped_tensor(self, tensor_name: str) -> str:
        """Get the mapped tensor name, following the chain."""
        source = tensor_name
        while source in self.tensor_mapping:
            source = self.tensor_mapping[source]
        return source
