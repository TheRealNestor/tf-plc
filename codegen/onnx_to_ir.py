"""
ONNX to Intermediate Representation (IR) Transformation Module
"""

from .types import *
from .onnx_model import ONNXModel
from typing import Dict, List
from collections import deque, defaultdict
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResolvedTensor:
    """Fully resolved tensor information (all static)"""

    name: str
    shape: Tuple[int, ...]
    dtype: Optional[str]
    size: int
    value: Optional[np.ndarray]  # None if not a constant/weight
    is_weight: bool


def resolve_layer_tensors(layer_dict: Dict, analyzer: ONNXModel) -> Dict:
    """
    Enrich layer dict with fully resolved tensor information.

    This is the ONLY place where analyzer is used - after this,
    extraction is purely dict-to-IR transformation.
    """
    resolved_inputs = []

    for inp_name in layer_dict["inputs"]:
        tensor_info = analyzer.tensor_info.get(inp_name, {})
        shape = tensor_info.get("shape", ())

        # Check if this is a direct weight/initializer
        is_weight = inp_name in analyzer.weights
        weight_value = analyzer.weights.get(inp_name)

        # For quantized models: check if this tensor is produced by DequantizeLinear
        # If so, trace back to find the actual quantized weight
        if not is_weight and weight_value is None:
            # Check if this is the output of a DequantizeLinear layer
            dequant_weight = _try_get_dequantized_weight(inp_name, analyzer)
            if dequant_weight is not None:
                is_weight = True
                weight_value = dequant_weight
                logger.debug(f"Resolved dequantized weight for {inp_name}")

        if is_weight:
            # Weights must be fully static
            if any(isinstance(dim, str) or dim is None for dim in shape):
                raise ValueError(
                    f"Symbolic dimension in weight tensor {inp_name}: {shape}"
                )
            static_shape = tuple(shape)
            size = int(np.prod(shape)) if shape else 0
        else:
            # Data tensors: skip symbolic/batch dimensions when calculating size
            static_dims = [d for d in shape if isinstance(d, int) and d > 0]
            static_shape = tuple(static_dims) if static_dims else ()
            size = int(np.prod(static_dims)) if static_dims else 0

        resolved_inputs.append(
            ResolvedTensor(
                name=inp_name,
                shape=static_shape,
                dtype=tensor_info.get("onnx_type"),
                size=size,
                value=weight_value,
                is_weight=is_weight,
            )
        )

    resolved_outputs = []
    for out_name in layer_dict["outputs"]:
        tensor_info = analyzer.tensor_info.get(out_name, {})
        shape = tensor_info.get("shape", ())

        # Skip symbolic dimensions
        static_dims = [d for d in shape if isinstance(d, int) and d > 0]
        static_shape = tuple(static_dims) if static_dims else ()
        size = int(np.prod(static_dims)) if static_dims else 0

        resolved_outputs.append(
            ResolvedTensor(
                name=out_name,
                shape=static_shape,
                dtype=tensor_info.get("onnx_type"),
                size=size,
                value=None,
                is_weight=False,
            )
        )

    return {
        **layer_dict,
        "resolved_inputs": resolved_inputs,
        "resolved_outputs": resolved_outputs,
    }


def _try_get_dequantized_weight(
    tensor_name: str, analyzer: ONNXModel
) -> Optional[np.ndarray]:
    """
    Try to resolve a dequantized weight tensor.

    In quantized models, weights are stored as:
      QuantizedWeight -> DequantizeLinear -> UsableWeight

    This function traces back from a tensor name to find if it's the output
    of a DequantizeLinear layer, and if so, dequantizes the weight.

    Args:
        tensor_name: Name of the tensor (potentially a DequantizeLinear output)
        analyzer: ONNX model analyzer

    Returns:
        Dequantized weight array, or None if not a dequantized weight
    """
    # Find the layer that produces this tensor
    producer_layer = None
    for layer in analyzer.layers:
        if tensor_name in layer["outputs"]:
            producer_layer = layer
            break

    if producer_layer is None or producer_layer["op_type"] != "DequantizeLinear":
        return None

    # DequantizeLinear has inputs: [quantized_data, scale, zero_point]
    inputs = producer_layer["inputs"]
    if len(inputs) < 3:
        logger.warning(
            f"DequantizeLinear layer {producer_layer['name']} has < 3 inputs"
        )
        return None

    quantized_tensor_name = inputs[0]
    scale_name = inputs[1]
    zero_point_name = inputs[2]

    # Get the quantized weight, scale, and zero_point
    quantized_weight = analyzer.weights.get(quantized_tensor_name)
    scale = analyzer.weights.get(scale_name)
    zero_point = analyzer.weights.get(zero_point_name)

    if quantized_weight is None:
        logger.debug(f"Quantized tensor {quantized_tensor_name} not found in weights")
        return None

    if scale is None or zero_point is None:
        logger.warning(
            f"Missing scale or zero_point for DequantizeLinear: "
            f"scale={scale_name}, zero_point={zero_point_name}"
        )
        return None

    # Dequantize: real_value = scale * (quantized_value - zero_point)
    try:
        # Handle both scalar and per-channel quantization
        if scale.ndim == 0:
            # Scalar quantization
            dequantized = scale * (
                quantized_weight.astype(np.float32) - zero_point.astype(np.float32)
            )
        else:
            # Per-channel quantization (scale and zero_point are vectors)
            # Broadcast along the appropriate axis
            axis = producer_layer.get("attributes", {}).get("axis", 1)

            # Reshape scale and zero_point for broadcasting
            shape = [1] * quantized_weight.ndim
            shape[axis] = -1
            scale_reshaped = scale.reshape(shape)
            zero_point_reshaped = zero_point.reshape(shape)

            dequantized = scale_reshaped * (
                quantized_weight.astype(np.float32)
                - zero_point_reshaped.astype(np.float32)
            )

        logger.debug(
            f"Dequantized weight {quantized_tensor_name} -> {tensor_name}: "
            f"shape={dequantized.shape}, dtype={dequantized.dtype}"
        )
        return dequantized

    except Exception as e:
        logger.error(f"Failed to dequantize weight {quantized_tensor_name}: {e}")
        return None


def extract_activation_layer(layer: Dict, layer_id: int) -> ActivationLayer:
    """Extract activation layer from enriched dict"""
    inputs = layer["resolved_inputs"]
    outputs = layer["resolved_outputs"]

    input_tensor = inputs[0]
    output_tensor = outputs[0]

    activation_type = ActivationType[layer["op_type"].upper()]

    return ActivationLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        activation=activation_type,
        input_size=input_tensor.size,
        output_size=output_tensor.size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in outputs),
        input_shape=input_tensor.shape,
        output_shape=output_tensor.shape,
        input_type=input_tensor.dtype,
        output_type=output_tensor.dtype,
    )


def extract_add_layer(layer: Dict, layer_id: int) -> AddLayer:
    """Extract Add layer from enriched dict"""
    inputs = layer["resolved_inputs"]
    outputs = layer["resolved_outputs"]

    input_tensor = inputs[0]
    bias_tensor = inputs[1]

    if not bias_tensor.is_weight or bias_tensor.value is None:
        raise ValueError(f"Add layer {layer_id} missing bias weight")

    return AddLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_tensor.size,
        output_size=outputs[0].size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in outputs),
        input_shape=input_tensor.shape,
        output_shape=outputs[0].shape,
        input_type=input_tensor.dtype,
        output_type=outputs[0].dtype,
        bias=bias_tensor.value,
    )


def extract_matmul_layer(layer: Dict, layer_id: int) -> MatMulLayer:
    """Extract MatMul layer from enriched dict"""
    inputs = layer["resolved_inputs"]
    outputs = layer["resolved_outputs"]

    data_tensor = inputs[0]
    weight_tensor = inputs[1]

    if not weight_tensor.is_weight or weight_tensor.value is None:
        raise ValueError(f"MatMul layer {layer_id} missing weight tensor")

    weights = weight_tensor.value
    input_size, output_size = weights.shape

    return MatMulLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        weights=weights,
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in outputs),
        input_shape=data_tensor.shape,
        output_shape=outputs[0].shape,
        input_type=data_tensor.dtype,
        output_type=outputs[0].dtype,
    )


def extract_gemm_layer(layer: Dict, layer_id: int) -> GemmLayer:
    """Extract Gemm layer from enriched dict"""
    inputs = layer["resolved_inputs"]
    outputs = layer["resolved_outputs"]
    attrs = layer.get("attributes", {})

    data_tensor = inputs[0]
    weight_tensor = inputs[1]

    if not weight_tensor.is_weight or weight_tensor.value is None:
        raise ValueError(f"Gemm layer {layer_id} missing weight tensor")

    weights = weight_tensor.value

    # Optional bias
    bias = None
    if len(inputs) > 2:
        bias_tensor = inputs[2]
        if bias_tensor.is_weight and bias_tensor.value is not None:
            bias = bias_tensor.value

    input_size, output_size = weights.shape

    return GemmLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        weights=weights,
        bias=bias,
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in outputs),
        input_shape=data_tensor.shape,
        output_shape=outputs[0].shape,
        input_type=data_tensor.dtype,
        output_type=outputs[0].dtype,
        alpha=attrs.get("alpha", 1.0),
        beta=attrs.get("beta", 1.0),
        transA=attrs.get("transA", False),
        transB=attrs.get("transB", False),
    )


def extract_fused_gemm_layer(layer: Dict, layer_id: int) -> FusedGemmLayer:
    """Extract FusedGemm layer from enriched dict"""
    inputs = layer["resolved_inputs"]
    outputs = layer["resolved_outputs"]
    attrs = layer.get("attributes", {})

    data_tensor = inputs[0]
    weight_tensor = inputs[1]

    if not weight_tensor.is_weight or weight_tensor.value is None:
        raise ValueError(f"FusedGemm layer {layer_id} missing weight tensor")

    weights = weight_tensor.value

    # Optional bias
    bias = None
    if len(inputs) > 2:
        bias_tensor = inputs[2]
        if bias_tensor.is_weight and bias_tensor.value is not None:
            bias = bias_tensor.value

    # Activation is required for FusedGemm
    if "activation" not in attrs:
        raise ValueError(
            f"FusedGemm layer {layer_id} missing required activation attribute"
        )

    input_size, output_size = weights.shape
    activation_type = ActivationType[attrs["activation"].upper()]

    return FusedGemmLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        weights=weights,
        bias=bias,
        activation=activation_type,
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in outputs),
        input_shape=data_tensor.shape,
        output_shape=outputs[0].shape,
        input_type=data_tensor.dtype,
        output_type=outputs[0].dtype,
        alpha=attrs.get("alpha", 1.0),
        beta=attrs.get("beta", 1.0),
        transA=attrs.get("transA", False),
        transB=attrs.get("transB", False),
    )


def extract_reshape_layer(layer: Dict, layer_id: int) -> ReshapeLayer:
    """Extract Reshape layer from enriched dict"""
    inputs = layer["resolved_inputs"]
    outputs = layer["resolved_outputs"]

    data_tensor = inputs[0]
    shape_tensor = inputs[1]

    if not shape_tensor.is_weight or shape_tensor.value is None:
        raise ValueError(f"Reshape layer {layer_id}: Target shape must be a constant")

    target_shape = tuple(int(dim) for dim in shape_tensor.value)
    input_size = data_tensor.size

    # Handle -1 in target shape (infer dimension)
    if -1 in target_shape:
        known_dims = [d for d in target_shape if d > 0]
        known_prod = int(np.prod(known_dims)) if known_dims else 1

        if known_prod == 0:
            raise ValueError(
                f"Reshape layer {layer_id}: Invalid target shape {target_shape}"
            )

        inferred = input_size // known_prod
        target_shape = tuple(inferred if d == -1 else d for d in target_shape)

    output_size = int(np.prod(target_shape))

    if output_size != input_size:
        raise ValueError(
            f"Reshape layer {layer_id}: Size mismatch - "
            f"input size {input_size} != output size {output_size}"
        )

    return ReshapeLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in outputs),
        input_shape=data_tensor.shape,
        output_shape=target_shape,
        input_type=data_tensor.dtype,
        output_type=outputs[0].dtype,
    )


# def extract_quantize_linear_layer(layer: Dict, layer_id: int) -> QuantizeLinearLayer:
#     """Extract QuantizeLinear layer from enriched dict"""
#     inputs = layer["resolved_inputs"]
#     outputs = layer["resolved_outputs"]
#     attrs = layer.get("attributes", {})

#     data_tensor = inputs[0]
#     scale_tensor = inputs[1]
#     zero_point_tensor = inputs[2]

#     if not scale_tensor.is_weight or scale_tensor.value is None:
#         raise ValueError(f"QuantizeLinear layer {layer_id} missing scale")
#     if not zero_point_tensor.is_weight or zero_point_tensor.value is None:
#         raise ValueError(f"QuantizeLinear layer {layer_id} missing zero_point")

#     return QuantizeLinearLayer(
#         layer_id=layer_id,
#         name=layer["name"],
#         op_type=layer["op_type"],
#         input_size=data_tensor.size,
#         output_size=outputs[0].size,
#         inputs=tuple(t.name for t in inputs),
#         outputs=tuple(t.name for t in outputs),
#         input_shape=data_tensor.shape,
#         output_shape=outputs[0].shape,
#         input_type=data_tensor.dtype,
#         output_type=outputs[0].dtype,
#         scale_name=scale_tensor.name,
#         zero_point_name=zero_point_tensor.name,
#         axis=attrs.get("axis"),
#     )


# def extract_dequantize_linear_layer(
#     layer: Dict, layer_id: int
# ) -> DequantizeLinearLayer:
#     """Extract DequantizeLinear layer from enriched dict"""
#     inputs = layer["resolved_inputs"]
#     outputs = layer["resolved_outputs"]
#     attrs = layer.get("attributes", {})

#     data_tensor = inputs[0]
#     scale_tensor = inputs[1]
#     zero_point_tensor = inputs[2]

#     if not scale_tensor.is_weight or scale_tensor.value is None:
#         raise ValueError(f"DequantizeLinear layer {layer_id} missing scale")
#     if not zero_point_tensor.is_weight or zero_point_tensor.value is None:
#         raise ValueError(f"DequantizeLinear layer {layer_id} missing zero_point")

#     return DequantizeLinearLayer(
#         layer_id=layer_id,
#         name=layer["name"],
#         op_type=layer["op_type"],
#         input_size=data_tensor.size,
#         output_size=outputs[0].size,
#         inputs=tuple(t.name for t in inputs),
#         outputs=tuple(t.name for t in outputs),
#         input_shape=data_tensor.shape,
#         output_shape=outputs[0].shape,
#         input_type=data_tensor.dtype,
#         output_type=outputs[0].dtype,
#         scale_name=scale_tensor.name,
#         zero_point_name=zero_point_tensor.name,
#         axis=attrs.get("axis"),
#     )


LAYER_EXTRACTORS = {
    "MatMul": extract_matmul_layer,
    "Add": extract_add_layer,
    "Gemm": extract_gemm_layer,
    "FusedGemm": extract_fused_gemm_layer,
    "Relu": extract_activation_layer,
    "Sigmoid": extract_activation_layer,
    "Tanh": extract_activation_layer,
    "Softmax": extract_activation_layer,
    "Reshape": extract_reshape_layer,
    # "QuantizeLinear": extract_quantize_linear_layer,
    # "DequantizeLinear": extract_dequantize_linear_layer,
}


def topological_sort(
    layers: Dict[str, BaseLayer],
    tensor_producers: Dict[str, str],
    tensor_consumers: Dict[str, List[str]],
    input_tensors: Tuple[str, ...],
) -> List[str]:
    """
    Perform topological sort on the layer graph using Kahn's algorithm.

    Returns list of layer names in execution order.
    """
    adj_list = defaultdict(list)
    in_degree = {name: 0 for name in layers.keys()}

    for layer_name, layer in layers.items():
        for input_tensor in layer.inputs:
            # Skip network inputs (they don't have producers)
            if input_tensor in input_tensors:
                continue

            if input_tensor in tensor_producers:
                producer = tensor_producers[input_tensor]
                if producer != layer_name:  # Avoid self-loops
                    adj_list[producer].append(layer_name)
                    in_degree[layer_name] += 1

    # Start with layers that have no dependencies
    queue = deque([name for name, degree in in_degree.items() if degree == 0])
    sorted_order = []

    while queue:
        current = queue.popleft()
        sorted_order.append(current)

        # Reduce in-degree for dependent layers
        for neighbor in adj_list[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles
    if len(sorted_order) != len(layers):
        missing = set(layers.keys()) - set(sorted_order)
        raise ValueError(f"Cycle detected in layer graph: {missing}")

    return sorted_order


def onnx_to_ir(analyzer: ONNXModel) -> NetworkIR:
    """Convert ONNX model to graph-based intermediate representation."""
    logger.info("Converting ONNX model to IR...")

    input_info, output_info = analyzer.get_input_output_info()
    input_tensor_names = tuple(input_info.get("names", []))
    output_tensor_names = tuple(output_info.get("names", []))

    # Phase 1: Resolve all tensors (inputs/outputs) for each layer
    # This includes shapes, dtypes, constant values for weights, etc
    onnx_layers = analyzer.analyze_layers()
    enriched_layers = []
    for layer_dict in onnx_layers:
        try:
            enriched = resolve_layer_tensors(layer_dict, analyzer)
            enriched_layers.append(enriched)
        except Exception as e:
            logger.error(f"Failed to resolve layer {layer_dict['name']}: {e}")
            raise

    # Phase 2: Propagate dtypes through the graph
    # ONNX shape inference doesn't always populate dtype information for all intermediate layers,
    # especially for fused layers, reshape layers, etc
    # This pass updates the resolved tensor dtypes by propagating known dtypes through the graph,
    # ensuring every tensor has a dtype before code generation (phase 3)

    tensor_dtypes: Dict[str, str] = {}

    # Seed with network inputs - these define the interface types for the PLC
    # IMPORTANT: Use the original input dtypes, not any quantized versions
    for inp_name, dtype in zip(input_info["names"], input_info["dtypes"]):
        tensor_dtypes[inp_name] = dtype
        logger.debug(f"Network input {inp_name} has dtype {dtype}")

    # Forward propagate dtypes through computation graph
    for enriched in enriched_layers:
        op_type = enriched["op_type"]

        # For quantization layers, don't propagate their output dtype
        # These are compile-time only and shouldn't affect PLC inference types
        if op_type in ["QuantizeLinear", "DequantizeLinear"]:
            # Still need to track what they produce, but use the INPUT dtype
            # This preserves the floating-point nature through the graph
            for resolved_input in enriched["resolved_inputs"]:
                if not resolved_input.is_weight and resolved_input.dtype is not None:
                    # Use the input dtype for outputs (e.g., FLOAT stays FLOAT)
                    for out_name in enriched["outputs"]:
                        tensor_dtypes[out_name] = resolved_input.dtype
                        logger.debug(
                            f"Quantization layer {enriched['name']}: "
                            f"preserving dtype {resolved_input.dtype} for {out_name}"
                        )
                    break
            continue

        # Update input dtypes from known tensor_dtypes
        updated_inputs = []
        for resolved_input in enriched["resolved_inputs"]:
            dtype = resolved_input.dtype
            if dtype is None and resolved_input.name in tensor_dtypes:
                dtype = tensor_dtypes[resolved_input.name]

            updated_inputs.append(
                ResolvedTensor(
                    name=resolved_input.name,
                    shape=resolved_input.shape,
                    dtype=dtype,
                    size=resolved_input.size,
                    value=resolved_input.value,
                    is_weight=resolved_input.is_weight,
                )
            )
        enriched["resolved_inputs"] = updated_inputs

        # Infer output dtype from first data input
        first_data_dtype = None
        for inp in updated_inputs:
            if not inp.is_weight and inp.dtype is not None:
                first_data_dtype = inp.dtype
                break

        # Update output dtypes
        updated_outputs = []
        for resolved_output in enriched["resolved_outputs"]:
            output_dtype = resolved_output.dtype or first_data_dtype

            updated_outputs.append(
                ResolvedTensor(
                    name=resolved_output.name,
                    shape=resolved_output.shape,
                    dtype=output_dtype,
                    size=resolved_output.size,
                    value=resolved_output.value,
                    is_weight=resolved_output.is_weight,
                )
            )

            if output_dtype is not None:
                tensor_dtypes[resolved_output.name] = output_dtype

        enriched["resolved_outputs"] = updated_outputs

    # Phase 3: Extract layers
    layers_dict = {}
    tensor_producers = {}
    tensor_consumers = {}

    for layer_id, layer_dict in enumerate(enriched_layers):
        op_type = layer_dict["op_type"]

        if op_type not in LAYER_EXTRACTORS:
            logger.warning(f"Unsupported ONNX layer type: {op_type}, skipping")
            continue

        try:
            extractor = LAYER_EXTRACTORS[op_type]
            layer = extractor(layer_dict, layer_id)

            layers_dict[layer.name] = layer

            # Track tensor producers and consumers
            for output_tensor in layer.outputs:
                tensor_producers[output_tensor] = layer.name

            for input_tensor in layer.inputs:
                if input_tensor not in tensor_consumers:
                    tensor_consumers[input_tensor] = []
                tensor_consumers[input_tensor].append(layer.name)

            logger.debug(f"Extracted layer {layer_id}: {layer.name} ({op_type})")

        except Exception as e:
            logger.error(f"Failed to extract layer {layer_id} ({op_type}): {e}")
            raise

    # Phase 4: Topological sort
    execution_order = topological_sort(
        layers_dict, tensor_producers, tensor_consumers, input_tensor_names
    )

    logger.info(f"Created IR with {len(layers_dict)} layers in execution order")

    return NetworkIR(
        layers=layers_dict,
        execution_order=execution_order,
        tensor_producers=tensor_producers,
        tensor_consumers=tensor_consumers,
        input_tensors=input_tensor_names,
        output_tensors=output_tensor_names,
    )
