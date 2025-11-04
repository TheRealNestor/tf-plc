"""
ONNX to Intermediate Representation (IR) Transformation Module
"""

# import everything from types.py
from .types import *
from typing import Dict, Optional
import numpy as np
from .onnx_model import ONNXModel


def extract_activation_layer(
    layer: Dict, layer_id: int, input_size: int, output_size: int
) -> ActivationLayer:
    activation_type = ActivationType[layer["op_type"].upper()]
    return ActivationLayer(
        layer_id=layer_id,
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
) -> Optional[AddLayer]:
    bias_tensor = None
    for input_name in layer["inputs"]:
        if input_name in weights:
            tensor = weights[input_name]
            if len(tensor.shape) == 1:
                bias_tensor = tensor
                break
    if bias_tensor is None:
        return None
    return AddLayer(
        layer_id=layer_id,
        bias=bias_tensor,
        input_size=input_size,
        output_size=output_size,
    )


def extract_matmul_layer(
    layer: Dict, layer_id: int, weights: Dict[str, np.ndarray]
) -> Optional[MatMulLayer]:
    weight_tensor = None
    for input_name in layer["inputs"]:
        if input_name in weights:
            tensor = weights[input_name]
            if len(tensor.shape) == 2:
                weight_tensor = tensor
                break
    if weight_tensor is None:
        return None
    input_size, output_size = weight_tensor.shape
    return MatMulLayer(
        layer_id=layer_id,
        weights=weight_tensor,
        input_size=input_size,
        output_size=output_size,
    )


def extract_gemm_layer(
    layer: Dict, layer_id: int, weights: Dict[str, np.ndarray]
) -> Optional[GemmLayer]:
    weight_tensor = None
    bias_tensor = None
    for input_name in layer["inputs"]:
        if input_name in weights:
            tensor = weights[input_name]
            if len(tensor.shape) == 2:
                weight_tensor = tensor
            elif len(tensor.shape) == 1:
                bias_tensor = tensor
    if weight_tensor is None:
        return None
    input_size, output_size = weight_tensor.shape
    alpha = layer.get("alpha", 1.0)
    beta = layer.get("beta", 1.0)
    transA = layer.get("transA", False)
    transB = layer.get("transB", False)
    return GemmLayer(
        layer_id=layer_id,
        weights=weight_tensor,
        bias=bias_tensor,
        input_size=input_size,
        output_size=output_size,
        alpha=alpha,
        beta=beta,
        transA=transA,
        transB=transB,
    )


def extract_fused_gemm_layer(
    layer: Dict, layer_id: int, weights: Dict[str, np.ndarray]
) -> Optional[FusedGemmLayer]:
    # Similar to GemmLayer, but expects an activation field
    weight_tensor = None
    bias_tensor = None
    for input_name in layer["inputs"]:
        if input_name in weights:
            tensor = weights[input_name]
            if len(tensor.shape) == 2:
                weight_tensor = tensor
            elif len(tensor.shape) == 1:
                bias_tensor = tensor
    if weight_tensor is None or "activation" not in layer:
        return None
    input_size, output_size = weight_tensor.shape
    activation = ActivationType[layer["activation"].upper()]
    alpha = layer.get("alpha", 1.0)
    beta = layer.get("beta", 1.0)
    transA = layer.get("transA", False)
    transB = layer.get("transB", False)
    return FusedGemmLayer(
        layer_id=layer_id,
        weights=weight_tensor,
        bias=bias_tensor,
        activation=activation,
        input_size=input_size,
        output_size=output_size,
        alpha=alpha,
        beta=beta,
        transA=transA,
        transB=transB,
    )


def extract_quantize_linear_layer(layer: Dict, layer_id: int) -> QuantizeLinearLayer:
    return QuantizeLinearLayer(
        layer_id=layer_id,
        input_name=layer["inputs"][0],
        scale_name=layer["inputs"][1],
        zero_point_name=layer["inputs"][2],
        output_name=layer["outputs"][0],
        axis=layer.get("axis"),
    )


def extract_dequantize_linear_layer(
    layer: Dict, layer_id: int
) -> DequantizeLinearLayer:
    return DequantizeLinearLayer(
        layer_id=layer_id,
        input_name=layer["inputs"][0],
        scale_name=layer["inputs"][1],
        zero_point_name=layer["inputs"][2],
        output_name=layer["outputs"][0],
        axis=layer.get("axis"),
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
}


def onnx_to_ir(analyzer: ONNXModel) -> NetworkIR:
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

    ir_layers = []
    layer_id = 0
    idx = 0
    layers = analyzer.layers

    while idx < len(layers):
        layer_dict = layers[idx]
        op_type = layer_dict["op_type"]
        extractor = LAYER_EXTRACTORS.get(op_type)
        layer = None
        consumed = 1 # ONNX layers are directly mapped one-to-one in the IR. Not combining layers. 

        if extractor:
            if op_type in ["MatMul"]:
                layer = extractor(layer_dict, layer_id, analyzer.weights)
            elif op_type in ["Add"]:
                prev_layer = ir_layers[-1] if ir_layers else None
                input_size = prev_layer.output_size if prev_layer else None
                output_size = input_size
                layer = extractor(
                    layer_dict, layer_id, analyzer.weights, input_size, output_size
                )
            elif op_type in ["Gemm"]:
                layer = extractor(layer_dict, layer_id, analyzer.weights)
            elif op_type in ["FusedGemm"]:
                layer = extractor(layer_dict, layer_id, analyzer.weights)
            elif op_type in ["Relu", "Sigmoid", "Tanh", "Softmax"]:
                prev_layer = ir_layers[-1] if ir_layers else None
                input_size = prev_layer.output_size if prev_layer else None
                output_size = input_size
                layer = extractor(layer_dict, layer_id, input_size, output_size)
            elif op_type in ["QuantizeLinear"]:
                layer = extractor(layer_dict, layer_id)
            elif op_type in ["DequantizeLinear"]:
                layer = extractor(layer_dict, layer_id)
            else:
                pass

            if layer:
                ir_layers.append(layer)
                layer_id += 1
        else:
            print(f"[WARNING] Unsupported ONNX layer: {op_type}")

        idx += consumed

    return NetworkIR(
        input_size=input_size, output_size=output_size, layers=tuple(ir_layers)
    )
