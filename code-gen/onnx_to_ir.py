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


def extract_common_layer_info(
    layer: Dict, layer_id: int
) -> Dict:
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
    type_info = {}

    if hasattr(analyzer, 'tensor_info') and analyzer.tensor_info:
        # Get input tensor info (first input is usually the data tensor)
        if layer.get('inputs'):
            input_name = layer['inputs'][0]
            if input_name in analyzer.tensor_info:
                info = analyzer.tensor_info[input_name]
                shape = info.get('shape', [])
                if shape:
                    type_info.update({
                        'input_type': info.get('dtype'),
                        'input_shape': tuple(shape) if shape else None,
                    })

        # Get output tensor info
        if layer.get('outputs'):
            output_name = layer['outputs'][0]
            if output_name in analyzer.tensor_info:
                info = analyzer.tensor_info[output_name]
                shape = info.get('shape', [])
                if shape:
                    type_info.update({
                        'output_type': info.get('dtype'),
                        'output_shape': tuple(shape) if shape else None,
                    })

    return type_info


def create_layer_base(layer: Dict, layer_id: int, analyzer: ONNXModel) -> Dict:
    """Create base layer info with common fields populated"""
    common_info = extract_common_layer_info(layer, layer_id)
    type_info = extract_type_info(layer, analyzer)
    return {**common_info, **type_info}


def calculate_sizes_from_tensor_info(layer: Dict, analyzer: ONNXModel) -> Tuple[int, int]:
    """Calculate input/output sizes from tensor info"""
    input_size = output_size = 1  
    
    if hasattr(analyzer, 'tensor_info') and analyzer.tensor_info and layer.get('inputs'):
        input_info = analyzer.tensor_info.get(layer['inputs'][0])
        if input_info and input_info.get('shape'):
            # Filter out dynamic dimensions (strings, -1, 0)
            shape = input_info['shape']
            static_dims = [d for d in shape if isinstance(d, int) and d > 0]
            input_size = int(np.prod(static_dims)) if static_dims else 1
            output_size = input_size
    

    return input_size, output_size


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


def extract_add_layer(layer: Dict, layer_id: int, weights: Dict[str, np.ndarray], input_size: int, output_size: int, analyzer: ONNXModel) -> AddLayer:
    base_info = create_layer_base(layer, layer_id, analyzer)

    bias_tensor = next(
        (weights[name] for name in layer["inputs"]
         if name in weights and len(weights[name].shape) == 1),
        None
    )

    if bias_tensor is None:
        logger.error(f"Add layer {layer_id} missing required bias tensor")
        raise ValueError(f"Add layer {layer_id} missing required bias tensor")

    return AddLayer(**base_info, bias=bias_tensor, input_size=input_size, output_size=output_size)


def extract_matmul_layer(layer: Dict, layer_id: int, weights: Dict[str, np.ndarray], analyzer: ONNXModel) -> MatMulLayer:
    base_info = create_layer_base(layer, layer_id, analyzer)

    weight_tensor = next(
        (weights[name] for name in layer["inputs"]
         if name in weights and len(weights[name].shape) == 2),
        None
    )

    if weight_tensor is None:
        logger.error(f"MatMul layer {layer_id} missing required weight tensor")
        raise ValueError(
            f"MatMul layer {layer_id} missing required weight tensor")

    input_size, output_size = weight_tensor.shape
    return MatMulLayer(
        **base_info,
        weights=weight_tensor,  
        input_size=input_size,
        output_size=output_size
    )


def extract_gemm_layer(layer: Dict, layer_id: int, weights: Dict[str, np.ndarray], analyzer: ONNXModel) -> GemmLayer:
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
        raise ValueError(
            f"Gemm layer {layer_id} missing required weight tensor")

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
    base_info = create_layer_base(
        layer, layer_id, analyzer)  # Use create_layer_base

    # Find weight and bias tensors
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
        logger.error(f"FusedGemm layer {layer_id} missing required weight tensor or activation attribute")
        raise ValueError(
            f"FusedGemm layer {layer_id} missing required weight tensor or activation attribute"
        )

    input_size, output_size = weight_tensor.shape
    activation_type = ActivationType[attrs["activation"].upper()]

    return FusedGemmLayer(
        **base_info,  # Use base_info instead of common_info + type_info
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

def extract_quantize_linear_layer(layer: Dict, layer_id: int, analyzer: ONNXModel) -> QuantizeLinearLayer:
    base_info = create_layer_base(layer, layer_id, analyzer)
    input_size, output_size = calculate_sizes_from_tensor_info(layer, analyzer)
    attrs = layer.get("attributes", {})

    return QuantizeLinearLayer(
        **base_info,
        scale_name=layer["inputs"][1],
        zero_point_name=layer["inputs"][2],
        axis=attrs.get("axis"),
        input_size=input_size,
        output_size=output_size,
    )


def extract_dequantize_linear_layer(layer: Dict, layer_id: int, analyzer: ONNXModel) -> DequantizeLinearLayer:
    base_info = create_layer_base(layer, layer_id, analyzer)
    input_size, output_size = calculate_sizes_from_tensor_info(layer, analyzer)
    attrs = layer.get("attributes", {})

    return DequantizeLinearLayer(
        **base_info,
        scale_name=layer["inputs"][1],
        zero_point_name=layer["inputs"][2],
        axis=attrs.get("axis"),
        input_size=input_size,
        output_size=output_size,
    )


def extract_reshape_layer(layer: Dict, layer_id: int, analyzer: ONNXModel) -> ReshapeLayer:
    # Use create_layer_base like others
    base_info = create_layer_base(layer, layer_id, analyzer)

    shape_input_name = layer["inputs"][1]
    if shape_input_name in analyzer.weights:
        raw_shape = analyzer.weights[shape_input_name]
        target_shape = tuple(int(x) for x in raw_shape)
    else:
        target_shape = tuple()

    input_name = layer["inputs"][0]
    input_info = analyzer.input_info.get(input_name, None)
    if input_info:
        input_shape = tuple(input_info["shape"])
        input_size = int(np.prod([d for d in input_shape if d > 0]))
    else:
        input_shape = tuple()
        input_size = int(np.prod(target_shape)) if target_shape else 0

    # Handle -1 in target_shape (infer dimension)
    if target_shape and -1 in target_shape:
        known = [d for d in target_shape if d > 0]
        known_prod = int(np.prod(known)) if known else 1
        inferred = int(input_size // known_prod) if known_prod != 0 else 0
        target_shape = tuple(inferred if d == -1 else d for d in target_shape)

    output_shape = target_shape if target_shape else input_shape
    output_size = int(np.prod(output_shape)) if output_shape else input_size

    if output_size <= 0:
        logger.error(f"Invalid output size for Reshape layer {layer_id}: {output_size}")
        raise ValueError(
            f"Invalid output size for Reshape layer {layer_id}: {output_size}")

    return ReshapeLayer(
        **base_info,  # Use base_info instead of common_info + type_info
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
    analyzer._build_tensor_info()

    input_info = list(analyzer.input_info.values())[0]
    output_info = list(analyzer.output_info.values())[0]
    input_shape = input_info["shape"]
    output_shape = output_info["shape"]

    # Filter out dynamic dimensions (negative values or string placeholders)
    def get_static_shape(shape):
        """Extract only static (positive integer) dimensions from shape"""
        static_dims = []
        for dim in shape:
            if isinstance(dim, int) and dim > 0:
                static_dims.append(dim)
            elif isinstance(dim, str):
                # Skip string placeholders like 'unk__8'
                continue
            elif dim <= 0:
                # Skip batch dimensions (-1) and other dynamic dims
                continue
        return static_dims

    static_input_shape = get_static_shape(input_shape)
    static_output_shape = get_static_shape(output_shape)
    input_size = int(np.prod(static_input_shape)) if static_input_shape else 1
    output_size = int(np.prod(static_output_shape)) if static_output_shape else 1

    logger.debug(f"Input shape: {static_input_shape} -> static input shape -> size {input_size}")
    logger.debug(f"Output shape: {static_output_shape} -> static output shape -> size {output_size}")

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

    return NetworkIR(
        input_size=input_size, output_size=output_size, layers=tuple(ir_layers)
    )
