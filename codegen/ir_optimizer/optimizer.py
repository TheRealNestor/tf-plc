"""
Main IR optimizer that orchestrates optimization passes.
"""

import logging
from typing import List, Optional
from collections import defaultdict

from ..onnx_to_ir.converter import topological_sort
from ..types import NetworkIR
from .base_pass import OptimizationPass
from .passes import (
    RemoveIdentityPass,
    RemoveNoOpReshapePass,
    RemoveRedundantQuantPairPass,
    RemoveWeightDequantPass,
    FuseLinearActivationPass,
    BufferAllocationPass,
)

logger = logging.getLogger(__name__)

DEFAULT_PASSES: List[OptimizationPass] = [
    RemoveIdentityPass(),
    RemoveWeightDequantPass(),
    RemoveNoOpReshapePass(),
    RemoveRedundantQuantPairPass(),
    FuseLinearActivationPass(),
    BufferAllocationPass(),
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

    def _filter_removed_layers(self, removed_layers: set) -> dict:
        """Remove layers marked for deletion."""
        return {
            name: layer
            for name, layer in self.ir.layers.items()
            if name not in removed_layers
        }

    def _follow_tensor_mapping(self, tensor: str, tensor_mapping: dict) -> str:
        """
        Follow tensor mapping chain to find final tensor.

        Handles transitive mappings: A->B, B->C results in A->C
        """
        source = tensor
        while source in tensor_mapping:
            source = tensor_mapping[source]
        return source

    def _rewire_layer_inputs(self, layers: dict, tensor_mapping: dict) -> None:
        """Update layer inputs to follow tensor remapping."""
        for layer in layers.values():
            new_inputs = [
                self._follow_tensor_mapping(inp, tensor_mapping) for inp in layer.inputs
            ]

            # Only update if changed (frozen dataclass workaround)
            if new_inputs != list(layer.inputs):
                object.__setattr__(layer, "inputs", tuple(new_inputs))

    def _remap_network_outputs(self, tensor_mapping: dict) -> list:
        """Remap network output tensors and log changes."""
        new_outputs = []

        for out in self.ir.output_tensors:
            remapped = self._follow_tensor_mapping(out, tensor_mapping)
            new_outputs.append(remapped)

            if remapped != out:
                logger.info(f"Remapped network output: {out} -> {remapped}")

        return new_outputs

    def _rebuild_graph_structure(self, layers: dict) -> tuple[dict, dict]:
        """
        Rebuild tensor producer/consumer maps.
        
        Returns:
            (tensor_producers, tensor_consumers) tuple
        """
        tensor_producers = {}
        tensor_consumers = defaultdict(list)

        for layer in layers.values():
            for inp in layer.inputs:
                tensor_consumers[inp].append(layer.name)
            for out in layer.outputs:
                tensor_producers[out] = layer.name

        return tensor_producers, dict(tensor_consumers)

    def _renumber_layer_ids(self, layers: dict, execution_order: list) -> dict:
        """Renumber layer IDs to be sequential based on execution order."""
        new_layers = {}
        for new_id, layer_name in enumerate(execution_order):
            layer = layers[layer_name]
            # Update layer_id using frozen dataclass workaround
            object.__setattr__(layer, "layer_id", new_id)
            new_layers[layer_name] = layer
        return new_layers

    def _rebuild_ir(self, removed_layers: set, tensor_mapping: dict) -> NetworkIR:
        """Rebuild IR with removed layers and rewired tensors.
        
        Args:
            removed_layers: Set of layer names to remove.
            tensor_mapping: Dict mapping old tensor names to new tensor names.
        Returns:
            Rebuilt NetworkIR with layers removed and tensors rewired.
        """

        if not removed_layers:
            return self.ir

        # 1. Remove layers
        new_layers = self._filter_removed_layers(removed_layers)

        # 2. Rewire tensor references in remaining layers
        self._rewire_layer_inputs(new_layers, tensor_mapping)

        # 3. Remap network outputs
        new_output_tensors = self._remap_network_outputs(tensor_mapping)

        # 4. Rebuild graph structure
        new_tensor_producers, new_tensor_consumers = self._rebuild_graph_structure(new_layers)

        # 5. Rebuild execution order
        new_execution_order = topological_sort(
            new_layers, new_tensor_producers, self.ir.input_tensors
        )

        # 6. Renumber layer IDs sequentially
        new_layers = self._renumber_layer_ids(new_layers, new_execution_order)

        return NetworkIR(
            layers=new_layers,
            execution_order=new_execution_order,
            tensor_producers=new_tensor_producers,
            tensor_consumers=new_tensor_consumers,
            input_tensors=self.ir.input_tensors,
            output_tensors=tuple(new_output_tensors),
        )
