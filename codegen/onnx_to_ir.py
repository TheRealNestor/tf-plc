"""
ONNX to Intermediate Representation (IR) Transformation Module
"""

# import everything from types.py
from .types import *
from typing import Dict
import numpy as np
from .onnx_model import ONNXModel

import logging

logger = logging.getLogger(__name__)


def resolve_static_dims(shape, tensor_name):
    """
    Extract static positive integer dimensions. We cannot feasibly handle dynamic/symbolic dims in PLC.
    Raise failure if NO static dims exist or shape is purely symbolic.
    """
    static = [d for d in shape if isinstance(d, int) and d > 0]

    if not static:
        raise ValueError(
            f"Cannot determine static size of tensor '{tensor_name}'. "
            f"Shape={shape}. "
            f"This model uses symbolic or dynamic dimensions "
            f"which PLC Structured Text cannot represent."
        )

    return static


def static_product(shape, tensor_name):
    """
    Compute product of static dims only.
    Raises an error if symbolic or unknown dims prevent determining a static size.
    """
    static = resolve_static_dims(shape, tensor_name)
    return int(np.prod(static))


def extract_common_layer_info(layer: Dict, layer_id: int) -> Dict:
    layer_name = layer.get("name") or f"{layer['op_type']}_{layer_id}"

    return {
        "layer_id": layer_id,
        "name": layer_name,
        "op_type": layer["op_type"],
        "inputs": tuple(layer.get("inputs", ())),
        "outputs": tuple(layer.get("outputs", ())),
    }


def extract_type_info(layer: Dict, analyzer: ONNXModel) -> Dict:
    """Extract type and shape information for layer inputs/outputs"""
    type_info: Dict[str, object] = {}

    if layer.get("inputs"):
        input_name = layer["inputs"][0]
        if input_name in analyzer.tensor_info:
            info = analyzer.tensor_info[input_name]
            shape = info.get("shape", [])
            if shape:
                type_info["input_type"] = info.get("onnx_type")
                type_info["input_shape"] = tuple(shape)
        else:
            logger.warning(
                f"Missing tensor_info for input '{input_name}' of layer "
                f"{layer.get('name', layer['op_type'])}"
            )

    if layer.get("outputs"):
        output_name = layer["outputs"][0]
        if output_name in analyzer.tensor_info:
            info = analyzer.tensor_info[output_name]
            shape = info.get("shape", [])
            if shape:
                type_info["output_type"] = info.get("onnx_type")
                type_info["output_shape"] = tuple(shape)
        else:
            logger.warning(
                f"Missing tensor_info for output '{output_name}' of layer "
                f"{layer.get('name', layer['op_type'])}"
            )

    return type_info


def create_layer_base(layer: Dict, layer_id: int, analyzer: ONNXModel) -> Dict:
    """Create base layer info with common fields populated"""
    common_info = extract_common_layer_info(layer, layer_id)
    type_info = extract_type_info(layer, analyzer)
    return {**common_info, **type_info}


def extract_activation_layer(
    layer: Dict, layer_id: int, input_size: int, output_size: int, analyzer: ONNXModel
) -> ActivationLayer:
    layer_base = create_layer_base(layer, layer_id, analyzer)
    activation_type = ActivationType[layer["op_type"].upper()]

    return ActivationLayer(
        **layer_base,
        activation=activation_type,
        input_size=input_size,
        output_size=output_size,
    )


def extract_add_layer(
    layer: Dict,
    layer_id: int,
    weights: Dict[str, np.ndarray],
    input_size: int,
    output_size: int,
    analyzer: ONNXModel,
) -> AddLayer:
    base_info = create_layer_base(layer, layer_id, analyzer)

    bias_tensor = next(
        (
            weights[name]
            for name in layer["inputs"]
            if name in weights and len(weights[name].shape) == 1
        ),
        None,
    )

    if bias_tensor is None:
        logger.error(f"Add layer {layer_id} missing required bias tensor")
        raise ValueError(f"Add layer {layer_id} missing required bias tensor")

    return AddLayer(
        **base_info, bias=bias_tensor, input_size=input_size, output_size=output_size
    )


def extract_matmul_layer(
    layer: Dict, layer_id: int, weights: Dict[str, np.ndarray], analyzer: ONNXModel
) -> MatMulLayer:
    base_info = create_layer_base(layer, layer_id, analyzer)

    weight_tensor = next(
        (
            weights[name]
            for name in layer["inputs"]
            if name in weights and len(weights[name].shape) == 2
        ),
        None,
    )

    if weight_tensor is None:
        logger.error(f"MatMul layer {layer_id} missing required weight tensor")
        raise ValueError(f"MatMul layer {layer_id} missing required weight tensor")

    input_size, output_size = weight_tensor.shape
    return MatMulLayer(
        **base_info,
        weights=weight_tensor,
        input_size=input_size,
        output_size=output_size,
    )


def extract_gemm_layer(
    layer: Dict, layer_id: int, weights: Dict[str, np.ndarray], analyzer: ONNXModel
) -> GemmLayer:
    base_info = create_layer_base(layer, layer_id, analyzer)

    weight_tensor = bias_tensor = None
    for name in layer["inputs"]:
        if name in weights:
            tensor = weights[name]
            if len(tensor.shape) == 2:
                weight_tensor = tensor
            elif len(tensor.shape) == 1:
                bias_tensor = tensor

    if weight_tensor is None:
        logger.error(f"Gemm layer {layer_id} missing required weight tensor")
        raise ValueError(f"Gemm layer {layer_id} missing required weight tensor")

    input_size, output_size = weight_tensor.shape
    attrs = layer.get("attributes", {})

    return GemmLayer(
        **base_info,
        weights=weight_tensor,
        bias=bias_tensor,
        input_size=input_size,
        output_size=output_size,
        alpha=attrs.get("alpha", 1.0),
        beta=attrs.get("beta", 1.0),
        transA=attrs.get("transA", False),
        transB=attrs.get("transB", False),
    )


def extract_fused_gemm_layer(
    layer: Dict, layer_id: int, weights: Dict[str, np.ndarray], analyzer: ONNXModel
) -> FusedGemmLayer:
    base_info = create_layer_base(layer, layer_id, analyzer)

    weight_tensor = bias_tensor = None
    for name in layer["inputs"]:
        if name in weights:
            tensor = weights[name]
            if len(tensor.shape) == 2:
                weight_tensor = tensor
            elif len(tensor.shape) == 1:
                bias_tensor = tensor

    attrs = layer.get("attributes", {})
    if weight_tensor is None or "activation" not in attrs:
        logger.error(
            f"FusedGemm layer {layer_id} missing required weight tensor or activation attribute"
        )
        raise ValueError(
            f"FusedGemm layer {layer_id} missing required weight tensor or activation attribute"
        )

    input_size, output_size = weight_tensor.shape
    activation_type = ActivationType[attrs["activation"].upper()]

    return FusedGemmLayer(
        **base_info,
        weights=weight_tensor,
        bias=bias_tensor,
        activation=activation_type,
        input_size=input_size,
        output_size=output_size,
        alpha=attrs.get("alpha", 1.0),
        beta=attrs.get("beta", 1.0),
        transA=attrs.get("transA", False),
        transB=attrs.get("transB", False),
    )


def extract_quantize_linear_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> QuantizeLinearLayer:
    base_info = create_layer_base(layer, layer_id, analyzer)

    # QuantizeLinear(input, scale, zero_point)
    input_name = layer["inputs"][0]

    if input_name not in analyzer.tensor_info:
        raise ValueError(
            f"QuantizeLinear {layer_id}: Missing tensor info for input '{input_name}'. "
            f"Cannot determine static shape."
        )

    input_shape = analyzer.tensor_info[input_name]["shape"]
    input_size = static_product(input_shape, input_name)

    # QuantizeLinear preserves the tensor shape
    output_size = input_size
    attrs = layer.get("attributes", {})

    return QuantizeLinearLayer(
        **base_info,
        scale_name=layer["inputs"][1],
        zero_point_name=layer["inputs"][2],
        axis=attrs.get("axis"),
        input_size=input_size,
        output_size=output_size,
    )


def extract_dequantize_linear_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> DequantizeLinearLayer:
    base_info = create_layer_base(layer, layer_id, analyzer)

    input_name = layer["inputs"][0]

    if input_name not in analyzer.tensor_info:
        raise ValueError(
            f"DequantizeLinear {layer_id}: Missing tensor info for input '{input_name}'. "
            f"Cannot determine static shape."
        )

    input_shape = analyzer.tensor_info[input_name]["shape"]
    input_size = static_product(input_shape, input_name)
    output_size = input_size

    attrs = layer.get("attributes", {})

    return DequantizeLinearLayer(
        **base_info,
        scale_name=layer["inputs"][1],
        zero_point_name=layer["inputs"][2],
        axis=attrs.get("axis"),
        input_size=input_size,
        output_size=output_size,
    )


def extract_reshape_layer(
    layer: Dict, layer_id: int, analyzer: ONNXModel
) -> ReshapeLayer:
    base_info = create_layer_base(layer, layer_id, analyzer)

    # The reshape target is usually a constant initializer
    shape_input_name = layer["inputs"][1]
    if shape_input_name in analyzer.weights:
        raw_target = analyzer.weights[shape_input_name]
        target_shape = tuple(int(dim) for dim in raw_target)
    else:
        raise ValueError(
            f"Reshape layer {layer_id}: Target shape tensor '{shape_input_name}' "
            f"is dynamic or missing. PLC code requires static reshape targets."
        )

    input_name = layer["inputs"][0]
    input_info = analyzer.tensor_info.get(input_name)
    if not input_info:
        raise ValueError(
            f"Reshape layer {layer_id}: Cannot determine input tensor info "
            f"for '{input_name}'."
        )

    input_shape = input_info["shape"]
    input_size = static_product(input_shape, input_name)

    if -1 in target_shape:
        known_dims = [d for d in target_shape if isinstance(d, int) and d > 0]
        known_prod = np.prod(known_dims) if known_dims else 1

        if known_prod == 0:
            raise ValueError(
                f"Reshape layer {layer_id}: Invalid known dims in target shape {target_shape}"
            )

        inferred = input_size // known_prod
        target_shape = tuple(inferred if d == -1 else d for d in target_shape)

    # Now validate the final target shape
    if any(d <= 0 for d in target_shape):
        raise ValueError(
            f"Reshape layer {layer_id}: Resolved target shape still contains "
            f"symbolic or invalid dims: {target_shape}"
        )

    output_size = int(np.prod(target_shape))

    if output_size != input_size:
        raise ValueError(
            f"Reshape layer {layer_id}: Cannot reshape tensor '{input_name}' "
            f"from shape {input_shape} (size={input_size}) "
            f"to target shape {target_shape} (size={output_size}). "
            f"ONNX requires reshape() to preserve total size, but the dimensions "
            f"do not match. This is not a symbolic-dim error; it is a size mismatch."
        )

    return ReshapeLayer(
        **base_info,
        input_size=input_size,
        output_size=output_size,
    )


LAYER_EXTRACTORS = {
    "MatMul": extract_matmul_layer,
    "Add": extract_add_layer,
    "Gemm": extract_gemm_layer,
    "FusedGemm": extract_fused_gemm_layer,
    "Relu": extract_activation_layer,
    "Sigmoid": extract_activation_layer,
    "Tanh": extract_activation_layer,
    "Softmax": extract_activation_layer,
    "QuantizeLinear": extract_quantize_linear_layer,
    "DequantizeLinear": extract_dequantize_linear_layer,
    "Reshape": extract_reshape_layer,
}


def onnx_to_ir(analyzer: ONNXModel) -> NetworkIR:
    """Convert ONNX model to intermediate representation"""

    logger.info("Converting ONNX model to IR...")

    # Ensure tensor info is built
    if not analyzer.tensor_info or not analyzer.input_info or not analyzer.output_info:
        logger.debug("tensor_info missing; rebuilding...")
        analyzer._build_tensor_info()

    logger.debug(f"Available tensor_info keys: {list(analyzer.tensor_info.keys())}")

    input_name = list(analyzer.input_info.keys())[0]
    output_name = list(analyzer.output_info.keys())[0]

    input_info = analyzer.input_info[input_name]
    output_info = analyzer.output_info[output_name]

    input_shape = input_info["shape"]
    output_shape = output_info["shape"]

    input_size = static_product(input_shape, input_name)
    output_size = static_product(output_shape, output_name)

    logger.debug(f"Static model input size = {input_size}, from shape {input_shape}")
    logger.debug(f"Static model output size = {output_size}, from shape {output_shape}")

    ir_layers = []
    layer_id = 0
    idx = 0

    while idx < len(analyzer.layers):
        layer_dict = analyzer.layers[idx]
        op_type = layer_dict["op_type"]
        extractor = LAYER_EXTRACTORS.get(op_type)
        layer = None
        consumed = 1

        if extractor:
            try:
                if op_type in ["MatMul"]:
                    layer = extractor(layer_dict, layer_id, analyzer.weights, analyzer)
                elif op_type in ["Add"]:
                    prev_layer = ir_layers[-1] if ir_layers else None
                    curr_input_size = (
                        prev_layer.output_size if prev_layer else input_size
                    )
                    curr_output_size = curr_input_size
                    layer = extractor(
                        layer_dict,
                        layer_id,
                        analyzer.weights,
                        curr_input_size,
                        curr_output_size,
                        analyzer,
                    )
                elif op_type in ["Gemm", "FusedGemm"]:
                    layer = extractor(layer_dict, layer_id, analyzer.weights, analyzer)
                elif op_type in ["Relu", "Sigmoid", "Tanh", "Softmax"]:
                    prev_layer = ir_layers[-1] if ir_layers else None
                    curr_input_size = (
                        prev_layer.output_size if prev_layer else input_size
                    )
                    curr_output_size = curr_input_size
                    layer = extractor(
                        layer_dict,
                        layer_id,
                        curr_input_size,
                        curr_output_size,
                        analyzer,
                    )
                elif op_type in ["QuantizeLinear", "DequantizeLinear", "Reshape"]:
                    layer = extractor(layer_dict, layer_id, analyzer)

                if layer:
                    ir_layers.append(layer)
                    layer_id += 1
            except Exception as e:
                logger.error(f"Failed to extract layer {layer_id} ({op_type}): {e}")
                raise
        else:
            logger.warning(f"Unsupported ONNX layer type: {op_type}")

        idx += consumed

    logger.info(f"Successfully converted {len(ir_layers)} layers to IR")

    network = NetworkIR(
        input_size=input_size, output_size=output_size, layers=tuple(ir_layers)
    )
    logger.info(network)

    return network
