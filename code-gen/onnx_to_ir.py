"""
ONNX to Intermediate Representation (IR) Transformation Module
"""

from typing import Dict, List, Tuple, Optional
from .types import DenseLayer, NetworkIR, ActivationType
import numpy as np
from .onnx_model import ONNXModel


def parse_layer_activation(
    layers: List[Dict], start_idx: int
) -> Tuple[ActivationType, int]:
    """
    Parse activation function following a MatMul/Gemm operation.
    Handles both direct activation and Add (bias) → activation patterns.

    Args:
        layers: List of ONNX layers
        start_idx: Index to start looking for activation

    Returns:
        Tuple of (activation_type, number_of_layers_consumed)
    """
    if start_idx >= len(layers):
        return (ActivationType.NONE, 0)

    next_layer = layers[start_idx]

    match next_layer["op_type"]:
        case "Relu":
            return (ActivationType.RELU, 1)
        case "Sigmoid":
            return (ActivationType.SIGMOID, 1)
        case "Tanh":
            return (ActivationType.TANH, 1)
        case "Softmax":
            return (ActivationType.SOFTMAX, 1)
        case "Add":
            if start_idx + 1 < len(layers):
                activation_layer = layers[start_idx + 1]
                match activation_layer["op_type"]:
                    case "Relu":
                        return (ActivationType.RELU, 2)
                    case "Sigmoid":
                        return (ActivationType.SIGMOID, 2)
                    case "Tanh":
                        return (ActivationType.TANH, 2)
                    case "Softmax":
                        return (ActivationType.SOFTMAX, 2)
                    case _:
                        return (ActivationType.NONE, 1)
            return (ActivationType.NONE, 1)
        case _:
            return (ActivationType.NONE, 0)


def extract_dense_layer(
    layer: Dict,
    layer_id: int,
    weights: Dict[str, np.ndarray],
    layers: List[Dict],
    layer_idx: int,
) -> Tuple[Optional[DenseLayer], int]:
    """
    Extract a dense layer from ONNX layer information.
    Handles MatMul + Add (bias) + Activation pattern from tf2onnx.

    Args:
        layer: ONNX layer dictionary
        layer_id: Numeric ID for this layer
        weights: Dictionary of weight tensors
        layers: All ONNX layers (for looking ahead at activation)
        layer_idx: Current index in layers list

    Returns:
        Tuple of (DenseLayer or None, number_of_layers_consumed)
    """
    match layer["op_type"]:
        case "Gemm" | "MatMul":
            weight_tensor = None

            for input_name in layer["inputs"]:
                if input_name in weights:
                    tensor = weights[input_name]
                    if len(tensor.shape) == 2:
                        weight_tensor = tensor
                        break

            if weight_tensor is None:
                return (None, 1)

            input_size, output_size = weight_tensor.shape

            bias_tensor = None
            consumed = 1

            if (
                layer_idx + 1 < len(layers)
                and layers[layer_idx + 1]["op_type"] == "Add"
            ):
                add_layer = layers[layer_idx + 1]
                for input_name in add_layer["inputs"]:
                    if input_name in weights:
                        tensor = weights[input_name]
                        if len(tensor.shape) == 1:
                            bias_tensor = tensor
                            break

            activation, activation_layers = parse_layer_activation(
                layers, layer_idx + 1
            )
            consumed += activation_layers

            return (
                DenseLayer(
                    layer_id=layer_id,
                    weights=weight_tensor,
                    bias=bias_tensor,
                    activation=activation,
                    input_size=input_size,
                    output_size=output_size,
                ),
                consumed,
            )
        case _:
            print(
                f"[WARNING] Unsupported ONNX layer encountered at index {layer_idx}: '{layer['op_type']}'"
            )
            print(f"         Layer details: {layer}")
            return (None, 1)


def onnx_to_ir(analyzer: ONNXModel) -> NetworkIR:
    """
    Transform ONNX model to intermediate representation.

    Args:
        analyzer: Analyzed ONNX model

    Returns:
        NetworkIR representation
    """
    input_info = list(analyzer.input_info.values())[0]
    output_info = list(analyzer.output_info.values())[0]

    input_shape = input_info["shape"]
    output_shape = output_info["shape"]

    input_size = (
        input_shape[1]
        if (input_shape[0] == -1 or input_shape[0] == 1)
        else input_shape[0]
    )
    output_size = (
        output_shape[1]
        if (output_shape[0] == -1 or output_shape[0] == 1)
        else output_shape[0]
    )

    dense_layers = []
    layer_id = 0
    idx = 0

    while idx < len(analyzer.layers):
        layer, consumed = extract_dense_layer(
            analyzer.layers[idx], layer_id, analyzer.weights, analyzer.layers, idx
        )

        if layer:
            dense_layers.append(layer)
            layer_id += 1

        idx += consumed

    return NetworkIR(
        input_size=input_size, output_size=output_size, layers=tuple(dense_layers)
    )
