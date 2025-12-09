"""
Buffer allocation optimization pass.

Analyzes tensor lifetimes and allocates reusable buffers using graph coloring.
"""

import logging
from typing import Dict, Set
from dataclasses import dataclass

import numpy as np

from ..base_pass import OptimizationPass
from ...types import NetworkIR

logger = logging.getLogger(__name__)


@dataclass
class BufferAllocation:
    """Represents a buffer allocation for a tensor."""

    buffer_name: str
    size: int
    dtype: str


class BufferAllocationPass(OptimizationPass):
    """Allocate reusable buffers for intermediate tensors using graph coloring."""

    def __init__(self):
        super().__init__()
        self.buffer_assignments: Dict[str, BufferAllocation] = {}

    def get_name(self) -> str:
        return "buffer_allocation"

    def optimize(self, ir: NetworkIR) -> None:
        """
        Allocate buffers using graph coloring.

        This pass only produces code generation hints (buffer_assignments).
        It does NOT modify the IR structure.

        Algorithm:
        1. Compute liveness intervals for each tensor
        2. Build interference graph (tensors that are live simultaneously)
        3. Color the graph (assign buffers to non-interfering tensors)
        """
        liveness = self._compute_liveness(ir)
        interference = self._build_interference_graph(liveness)
        self._color_graph(ir, interference)

    def _compute_liveness(self, ir: NetworkIR) -> Dict[str, tuple]:
        """Compute (start, end) indices for each tensor's lifetime."""
        liveness = {}

        for i, layer_name in enumerate(ir.execution_order):
            layer = ir.get_layer(layer_name)

            # Mark tensor as produced at this index
            for output_tensor in layer.outputs:
                if output_tensor not in liveness:
                    liveness[output_tensor] = [i, i]

            # Update last-use for input tensors
            for input_tensor in layer.inputs:
                if input_tensor in liveness:
                    liveness[input_tensor][1] = i

        return {k: tuple(v) for k, v in liveness.items()}

    def _build_interference_graph(
        self, liveness: Dict[str, tuple]
    ) -> Dict[str, Set[str]]:
        """Build graph of tensors that are live at the same time."""
        interference = {tensor: set() for tensor in liveness}

        tensors = list(liveness.keys())
        for i, t1 in enumerate(tensors):
            start1, end1 = liveness[t1]
            for t2 in tensors[i + 1 :]:
                start2, end2 = liveness[t2]

                # Check if intervals overlap
                if not (end1 < start2 or end2 < start1):
                    interference[t1].add(t2)
                    interference[t2].add(t1)

        return interference

    def _color_graph(
        self,
        ir: NetworkIR,
        interference: Dict[str, Set[str]],
    ) -> None:
        """Assign buffers using greedy graph coloring."""

        # Get tensor sizes
        tensor_sizes = {}
        for layer_name in ir.execution_order:
            layer = ir.get_layer(layer_name)
            if hasattr(layer, "output_shape") and layer.output_shape:
                for output_tensor in layer.outputs:
                    tensor_sizes[output_tensor] = int(np.prod(layer.output_shape))

        # Sort tensors by number of conflicts (most constrained first)
        sorted_tensors = sorted(
            interference.keys(), key=lambda t: len(interference[t]), reverse=True
        )

        # Greedy coloring
        buffer_colors: Dict[str, int] = {}
        buffer_sizes: Dict[int, int] = {}  # color -> max size needed

        for tensor in sorted_tensors:
            # Skip network inputs/outputs
            if tensor in ir.input_tensors or tensor in ir.output_tensors:
                continue

            # Find first color not used by neighbors
            neighbor_colors = {
                buffer_colors[n] for n in interference[tensor] if n in buffer_colors
            }

            color = 0
            while color in neighbor_colors:
                color += 1

            buffer_colors[tensor] = color

            # Track max size needed for this buffer
            size = tensor_sizes.get(tensor, 0)
            buffer_sizes[color] = max(buffer_sizes.get(color, 0), size)

        # Create buffer allocations (metadata only - doesn't modify IR)
        for tensor, color in buffer_colors.items():
            self.buffer_assignments[tensor] = BufferAllocation(
                buffer_name=f"buffer_{color}", size=buffer_sizes[color], dtype="REAL"
            )

        logger.info(
            f"Buffer allocation: {len(buffer_colors)} tensors -> {len(buffer_sizes)} buffers"
        )
