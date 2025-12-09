"""
Main ONNX to IR conversion orchestration.
"""

import numpy as np
import logging
from typing import Dict, List
from collections import deque, defaultdict

from ..types import NetworkIR, BaseLayer
from ..onnx_model import ONNXModel
from .tensor_resolution import TensorResolver, ResolvedTensor
from .shape_inference import infer_layer_shapes
from .layer_extractors import LAYER_EXTRACTORS

logger = logging.getLogger(__name__)


def topological_sort(
    layers: Dict[str, BaseLayer],
    tensor_producers: Dict[str, str],
    input_tensors: tuple,
) -> List[str]:
    """
    Perform topological sort on the layer graph using Kahn's algorithm.

    Args:
        layers: Dictionary of IR layer objects
        tensor_producers: Mapping of tensor names to producing layer names
        input_tensors: Network input tensor names

    Returns:
        List of layer names in execution order
    """
    adj_list = defaultdict(list)
    in_degree = {name: 0 for name in layers.keys()}

    for layer_name, layer in layers.items():
        for input_tensor in layer.inputs:
            if input_tensor in input_tensors:
                continue

            if input_tensor in tensor_producers:
                producer = tensor_producers[input_tensor]
                if producer != layer_name:
                    adj_list[producer].append(layer_name)
                    in_degree[layer_name] += 1

    queue = deque([name for name, degree in in_degree.items() if degree == 0])
    sorted_order = []

    while queue:
        current = queue.popleft()
        sorted_order.append(current)

        for neighbor in adj_list[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_order) != len(layers):
        missing = set(layers.keys()) - set(sorted_order)
        raise ValueError(f"Cycle detected in layer graph: {missing}")

    return sorted_order


def onnx_to_ir(analyzer: ONNXModel) -> NetworkIR:
    """
    Convert ONNX model to intermediate representation (IR).

    This creates a complete IR without optimization.
    Use IROptimizer for post-processing.
    """
    logger.info("Converting ONNX model to IR...")

    resolver = TensorResolver(analyzer)
    input_info, output_info = analyzer.get_input_output_info()
    input_tensors = tuple(input_info["names"])
    output_tensors = tuple(output_info["names"])

    layers: Dict[str, BaseLayer] = {}
    tensor_producers: Dict[str, str] = {}
    tensor_consumers: Dict[str, List[str]] = defaultdict(list)

    # Process each layer
    for layer_id, layer_dict in enumerate(analyzer.layers):

        enriched_layer = resolver.resolve_layer_tensors(layer_dict)
        _, output_shape = infer_layer_shapes(enriched_layer)

        for out_name in enriched_layer["outputs"]:
            resolver.store_inferred_shape(out_name, output_shape)

        if output_shape and enriched_layer["resolved_outputs"]:
            enriched_layer["resolved_outputs"] = [
                ResolvedTensor(
                    name=out.name,
                    shape=output_shape,
                    dtype=out.dtype,
                    size=int(np.prod(output_shape)) if output_shape else 0,
                    value=out.value,
                    is_weight=out.is_weight,
                )
                for out in enriched_layer["resolved_outputs"]
            ]

        op_type = enriched_layer["op_type"]
        if op_type in LAYER_EXTRACTORS:
            try:
                ir_layer = LAYER_EXTRACTORS[op_type](enriched_layer, layer_id, analyzer)
                layers[ir_layer.name] = ir_layer
                logger.debug(f"Extracted layer {layer_id}: {ir_layer.name} ({op_type})")

                # Track graph structure
                for inp in ir_layer.inputs:
                    tensor_consumers[inp].append(ir_layer.name)
                for out in ir_layer.outputs:
                    tensor_producers[out] = ir_layer.name

            except Exception as e:
                logger.error(f"Failed to extract layer {layer_id} ({op_type}): {e}")
                raise
        else:
            logger.warning(f"Unsupported layer type: {op_type}")

    # Sort layers
    execution_order = topological_sort(layers, tensor_producers, input_tensors)

    logger.info(f"Created IR with {len(layers)} layers in execution order")

    return NetworkIR(
        layers=layers,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
    )
