"""
Main IR optimizer that orchestrates optimization passes.
"""

import logging
from typing import List, Optional
from collections import defaultdict

from ..types import NetworkIR
from .base_pass import OptimizationPass
from .passes import (
    RemoveIdentityPass,
    RemoveNoOpReshapePass,
    FuseQuantDequantPass,
)

logger = logging.getLogger(__name__)


# Default optimization passes (order matters)
DEFAULT_PASSES = [
    RemoveIdentityPass(),
    FuseQuantDequantPass(),
    RemoveNoOpReshapePass(),
]


class IROptimizer:
    """Applies optimization passes to NetworkIR."""

    def __init__(self, ir: NetworkIR):
        self.ir = ir

    def optimize(self, passes: Optional[List[OptimizationPass]] = None) -> NetworkIR:
        """
        Apply optimization passes to the IR.

        Args:
            passes: List of pass instances to apply. If None, uses DEFAULT_PASSES.

        Returns:
            Optimized NetworkIR
        """
        if passes is None:
            passes = DEFAULT_PASSES

        logger.info(f"Applying {len(passes)} optimization passes...")

        # Collect all removals and mappings
        all_removed_layers = set()
        all_tensor_mappings = {}

        for pass_instance in passes:
            logger.info(f"Running pass: {pass_instance.get_name()}")
            pass_instance.run(self.ir)

            # Collect results
            all_removed_layers.update(pass_instance.removed_layers)
            all_tensor_mappings.update(pass_instance.tensor_mapping)

        # Rebuild IR with all optimizations applied
        return self._rebuild_ir(all_removed_layers, all_tensor_mappings)

    def _rebuild_ir(self, removed_layers: set, tensor_mapping: dict) -> NetworkIR:
        """Rebuild IR with removed layers and rewired tensors."""
        if not removed_layers:
            logger.info("No layers removed, returning original IR")
            return self.ir

        logger.info(f"Removing {len(removed_layers)} layers from IR...")

        # 1. Remove layers
        new_layers = {
            name: layer
            for name, layer in self.ir.layers.items()
            if name not in removed_layers
        }

        # 2. Rewire tensor references in remaining layers
        for layer in new_layers.values():
            # Remap inputs (follow mapping chain)
            new_inputs = []
            for inp in layer.inputs:
                source = inp
                while source in tensor_mapping:
                    source = tensor_mapping[source]
                new_inputs.append(source)

            # Update layer inputs if changed
            if new_inputs != list(layer.inputs):
                object.__setattr__(layer, "inputs", tuple(new_inputs))

        # 3. Remap network outputs
        new_output_tensors = []
        for out in self.ir.output_tensors:
            source = out
            while source in tensor_mapping:
                source = tensor_mapping[source]
            new_output_tensors.append(source)

            if source != out:
                logger.info(f"Remapped network output: {out} -> {source}")

        # 4. Rebuild graph structure
        new_tensor_producers = {}
        new_tensor_consumers = defaultdict(list)

        for layer in new_layers.values():
            for inp in layer.inputs:
                new_tensor_consumers[inp].append(layer.name)
            for out in layer.outputs:
                new_tensor_producers[out] = layer.name

        # 5. Rebuild execution order
        from ..onnx_to_ir.converter import topological_sort

        new_execution_order = topological_sort(
            new_layers, new_tensor_producers, self.ir.input_tensors
        )

        logger.info(
            f"Optimized IR: {len(self.ir.layers)} -> {len(new_layers)} layers "
            f"({len(removed_layers)} removed)"
        )

        return NetworkIR(
            layers=new_layers,
            execution_order=new_execution_order,
            tensor_producers=new_tensor_producers,
            tensor_consumers=new_tensor_consumers,
            input_tensors=self.ir.input_tensors,
            output_tensors=tuple(new_output_tensors),
        )
