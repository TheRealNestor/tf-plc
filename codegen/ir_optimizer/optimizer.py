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
    RemoveRedundantQuantPairPass,
    RemoveWeightDequantPass,
)

logger = logging.getLogger(__name__)

DEFAULT_PASSES: List[OptimizationPass] = [
    RemoveIdentityPass(),
    RemoveWeightDequantPass(),
    RemoveNoOpReshapePass(),
    RemoveRedundantQuantPairPass(),
]


class IROptimizer:
    """Applies optimization passes to NetworkIR."""

    def __init__(self, ir: NetworkIR, passes: Optional[List[OptimizationPass]] = None):
        self.ir = ir
        self.passes = passes if passes is not None else DEFAULT_PASSES

    def optimize(self) -> NetworkIR:
        """
        Apply optimization passes to the IR.

        Rebuilds IR between passes so each pass sees a clean, updated graph.

        Returns:
            Optimized NetworkIR
        """
        initial_layer_count = len(self.ir.layers)
        logger.info(f"Starting optimization with {initial_layer_count} layers")

        # Run each pass and rebuild IR between passes
        for pass_instance in self.passes:
            logger.info(f"Running pass: {pass_instance.get_name()}")
            pass_instance.optimize(self.ir)

            # Rebuild IR after each pass if it removed layers
            if pass_instance.removed_layers:
                self.ir = self._rebuild_ir(
                    pass_instance.removed_layers, pass_instance.tensor_mapping
                )

        final_layer_count = len(self.ir.layers)
        total_removed = initial_layer_count - final_layer_count

        logger.info(
            f"Optimization complete: {initial_layer_count} -> {final_layer_count} layers "
            f"({total_removed} removed)"
        )

        return self.ir

    def _rebuild_ir(self, removed_layers: set, tensor_mapping: dict) -> NetworkIR:
        """Rebuild IR with removed layers and rewired tensors."""
        if not removed_layers:
            return self.ir

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

        return NetworkIR(
            layers=new_layers,
            execution_order=new_execution_order,
            tensor_producers=new_tensor_producers,
            tensor_consumers=new_tensor_consumers,
            input_tensors=self.ir.input_tensors,
            output_tensors=tuple(new_output_tensors),
        )
