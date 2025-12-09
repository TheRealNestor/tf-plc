"""
Layer extraction functions.
Convert enriched layer dicts to IR layer objects.
"""

import numpy as np
import logging
from typing import Dict
from ..types import *
from .shape_inference import (
    infer_layer_shapes,
    get_feature_sizes,
    validate_inferred_shapes,
)
from .weight_utils import extract_quantized_weight, validate_weight_quantization
from ..onnx_model import ONNXModel

logger = logging.getLogger(__name__)


def extract_activation_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> ActivationLayer:
    """Extract activation layer."""
    inputs = layer["resolved_inputs"]
    outputs = layer["resolved_outputs"]
    activation_type = (
        layer["op_type"].upper()
        if layer["op_type"].upper() in ActivationType.__members__
        else "NONE"
    )

    return ActivationLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        activation=ActivationType[activation_type],
        input_size=inputs[0].size,
        output_size=outputs[0].size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in outputs),
        input_shape=inputs[0].shape,
        output_shape=outputs[0].shape,
        input_type=inputs[0].dtype,
        output_type=outputs[0].dtype,
    )


def extract_add_layer(layer: Dict, layer_id: int, analyzer: ONNXModel) -> AddLayer:
    """Extract Add layer (element-wise addition with optional constant)."""
    inputs = layer["resolved_inputs"]
    outputs = layer["resolved_outputs"]

    # Check if second input is a constant (bias/weight)
    bias = inputs[1].value if inputs[1].is_weight else None

    # For bias addition, only pass the tensor input
    # For element-wise, pass both tensor inputs
    layer_inputs = (
        (inputs[0].name,) if bias is not None else tuple(t.name for t in inputs)
    )

    return AddLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=inputs[0].size,
        output_size=outputs[0].size,
        inputs=layer_inputs,
        outputs=tuple(t.name for t in outputs),
        input_shape=inputs[0].shape,
        output_shape=outputs[0].shape,
        input_type=inputs[0].dtype,
        output_type=outputs[0].dtype,
        bias=bias,  # None for element-wise, array for bias addition
    )


def extract_matmul_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> MatMulLayer:
    """Extract MatMul layer."""
    inputs = layer["resolved_inputs"]

    input_shape, output_shape = infer_layer_shapes(layer)
    input_size, output_size = get_feature_sizes(input_shape, output_shape)

    validate_inferred_shapes(
        layer["name"], "MatMul", input_shape, output_shape, inputs[1].shape
    )

    weights, scale, zero_point = extract_quantized_weight(
        inputs[1].name, analyzer.layers, analyzer.weights
    )

    validate_weight_quantization(weights, scale, zero_point, input_size, output_size)

    return MatMulLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        weights=weights,
        weight_scale=scale,
        weight_zero_point=zero_point,
    )


def extract_gemm_layer(layer: Dict, layer_id: int, analyzer: ONNXModel) -> GemmLayer:
    """Extract Gemm layer."""
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})

    input_shape, output_shape = infer_layer_shapes(layer)
    input_size, output_size = get_feature_sizes(input_shape, output_shape)

    validate_inferred_shapes(
        layer["name"], "Gemm", input_shape, output_shape, inputs[1].shape
    )

    weights, scale, zero_point = extract_quantized_weight(
        inputs[1].name, analyzer.layers, analyzer.weights
    )

    validate_weight_quantization(weights, scale, zero_point, input_size, output_size)

    return GemmLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        weights=weights,
        bias=inputs[2].value if len(inputs) > 2 and inputs[2].is_weight else None,
        alpha=attrs.get("alpha", 1.0),
        beta=attrs.get("beta", 1.0),
        transA=attrs.get("transA", 0) == 1,
        transB=attrs.get("transB", 0) == 1,
        weight_scale=scale,
        weight_zero_point=zero_point,
    )


def extract_fused_gemm_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> FusedGemmLayer:
    """Extract FusedGemm layer."""
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})

    input_shape, output_shape = infer_layer_shapes(layer)
    input_size, output_size = get_feature_sizes(input_shape, output_shape)

    validate_inferred_shapes(
        layer["name"], "FusedGemm", input_shape, output_shape, inputs[1].shape
    )

    weights, scale, zero_point = extract_quantized_weight(
        inputs[1].name, analyzer.layers, analyzer.weights
    )

    validate_weight_quantization(weights, scale, zero_point, input_size, output_size)

    return FusedGemmLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        activation=ActivationType[attrs.get("activation", "RELU").upper()],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=input_shape,
        output_shape=output_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        weights=weights,
        bias=inputs[2].value if len(inputs) > 2 and inputs[2].is_weight else None,
        alpha=attrs.get("alpha", 1.0),
        beta=attrs.get("beta", 1.0),
        transA=attrs.get("transA", 0) == 1,
        transB=attrs.get("transB", 0) == 1,
        weight_scale=scale,
        weight_zero_point=zero_point,
    )


def extract_reshape_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> ReshapeLayer:
    """Extract Reshape layer."""
    inputs = layer["resolved_inputs"]

    if not inputs[1].is_weight or inputs[1].value is None:
        raise ValueError(f"Reshape layer {layer_id}: Target shape must be constant")

    target_shape = tuple(int(dim) for dim in inputs[1].value)
    input_size = inputs[0].size

    # Handle -1 in target shape
    if -1 in target_shape:
        known_dims = [d for d in target_shape if d > 0]
        known_prod = int(np.prod(known_dims)) if known_dims else 1
        inferred = input_size // known_prod
        target_shape = tuple(inferred if d == -1 else d for d in target_shape)

    output_size = int(np.prod(target_shape))

    if output_size != input_size:
        raise ValueError(
            f"Reshape {layer_id}: Size mismatch - "
            f"input {input_size} != output {output_size}"
        )

    return ReshapeLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=input_size,
        output_size=output_size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=inputs[0].shape,
        output_shape=target_shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
    )


def extract_quantize_linear_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> QuantizeLinearLayer:
    """Extract QuantizeLinear layer."""
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})

    if not inputs[1].is_weight or inputs[1].value is None:
        raise ValueError(f"QuantizeLinear {layer_id} missing scale")

    scale = inputs[1].value
    zero_point = (
        inputs[2].value
        if len(inputs) > 2 and inputs[2].value is not None
        else np.array([0])
    )

    return QuantizeLinearLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=inputs[0].size,
        output_size=layer["resolved_outputs"][0].size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=inputs[0].shape,
        output_shape=layer["resolved_outputs"][0].shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        scale=scale,
        zero_point=zero_point,
        axis=attrs.get("axis"),
    )


def extract_dequantize_linear_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> DequantizeLinearLayer:
    """Extract DequantizeLinear layer."""
    inputs = layer["resolved_inputs"]
    attrs = layer.get("attributes", {})

    if not inputs[1].is_weight or inputs[1].value is None:
        raise ValueError(f"DequantizeLinear {layer_id} missing scale")

    scale = inputs[1].value
    zero_point = (
        inputs[2].value
        if len(inputs) > 2 and inputs[2].value is not None
        else np.array([0])
    )

    return DequantizeLinearLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=inputs[0].size,
        output_size=layer["resolved_outputs"][0].size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in layer["resolved_outputs"]),
        input_shape=inputs[0].shape,
        output_shape=layer["resolved_outputs"][0].shape,
        input_type=inputs[0].dtype,
        output_type=layer["resolved_outputs"][0].dtype,
        scale=scale,
        zero_point=zero_point,
        axis=attrs.get("axis"),
    )


def extract_dropout_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> DropoutLayer:
    """
    Extract Dropout layer.

    Note: Dropout is only active during training. At inference time,
    it acts as an identity/pass-through operation.
    """
    inputs = layer["resolved_inputs"]
    outputs = layer["resolved_outputs"]
    attrs = layer.get("attributes", {})

    ratio = attrs.get("ratio", 0.5)

    return DropoutLayer(
        layer_id=layer_id,
        name=layer["name"],
        op_type=layer["op_type"],
        input_size=inputs[0].size,
        output_size=outputs[0].size,
        inputs=tuple(t.name for t in inputs),
        outputs=tuple(t.name for t in outputs),
        input_shape=inputs[0].shape,
        output_shape=outputs[0].shape,
        input_type=inputs[0].dtype,
        output_type=outputs[0].dtype,
        ratio=ratio,
    )


# Registry of layer extractors
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
    "QuantizeLinear": extract_quantize_linear_layer,
    "DequantizeLinear": extract_dequantize_linear_layer,
    "Dropout": extract_dropout_layer,
}
