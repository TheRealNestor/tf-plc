"""
Weight handling utilities for ONNX to IR conversion.
Normalizes quantized weights during IR construction.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def find_dequantize_producer(
    tensor_name: str, layers: list
) -> Optional[Dict[str, Any]]:
    """
    Find if a tensor is produced by DequantizeLinear.

    Returns:
        Layer dict if found, else None
    """
    for layer in layers:
        if tensor_name in layer["outputs"] and layer["op_type"] == "DequantizeLinear":
            return layer
    return None


def extract_quantized_weight(
    weight_tensor_name: str, layers: list, weights_dict: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract weight information, detecting if it comes from DequantizeLinear.

    This function normalizes quantized weights at IR construction time:
    - If weight comes from DequantizeLinear, extract the quantized form
    - If weight is float, return as-is

    Returns:
        (weights, scale, zero_point)
        - For float weights: (float_array, None, None)
        - For quantized weights: (int8_array, scale_array, zp_array)
    """
    # Check if this tensor is produced by DequantizeLinear
    producer = find_dequantize_producer(weight_tensor_name, layers)

    if producer is None:
        # Float weight - direct access
        weight_data = weights_dict.get(weight_tensor_name)
        if weight_data is None:
            raise ValueError(f"Weight tensor '{weight_tensor_name}' not found")
        return weight_data, None, None

    # Quantized weight - extract from DequantizeLinear inputs
    inputs = producer["inputs"]
    if len(inputs) < 3:
        logger.warning(
            f"DequantizeLinear has insufficient inputs: {inputs}. "
            f"Falling back to float weight."
        )
        weight_data = weights_dict.get(weight_tensor_name)
        return weight_data, None, None

    quantized_tensor_name = inputs[0]
    scale_name = inputs[1]
    zero_point_name = inputs[2]

    # Get the tensors
    quantized_weight = weights_dict.get(quantized_tensor_name)
    scale = weights_dict.get(scale_name)
    zero_point = weights_dict.get(zero_point_name)

    if quantized_weight is None:
        raise ValueError(f"Quantized weight '{quantized_tensor_name}' not found")

    if scale is None or zero_point is None:
        logger.warning(
            f"Missing scale/zero_point for '{weight_tensor_name}'. "
            f"Falling back to float weight."
        )
        weight_data = weights_dict.get(weight_tensor_name)
        return weight_data, None, None

    logger.info(
        f"Extracted quantized weight '{quantized_tensor_name}' "
        f"(dtype={quantized_weight.dtype}, scale_size={scale.size})"
    )

    return quantized_weight, scale, zero_point


def validate_weight_quantization(
    weights: np.ndarray,
    scale: Optional[np.ndarray],
    zero_point: Optional[np.ndarray],
    input_size: int,
    output_size: int,
) -> None:
    """
    Validate that quantization parameters are consistent with weight shape.

    Raises:
        ValueError: If quantization parameters are inconsistent
    """
    if scale is None and zero_point is None:
        # Float weights - no validation needed
        return

    if scale is None or zero_point is None:
        raise ValueError("Scale and zero_point must both be present or both be None")

    # Check weight dtype
    if weights.dtype not in [np.int8, np.uint8]:
        raise ValueError(
            f"Quantized weights must be int8 or uint8, got {weights.dtype}"
        )

    # Validate scale/zero_point sizes
    if scale.size == 1:
        # Per-tensor quantization
        if zero_point.size != 1:
            raise ValueError(
                f"Per-tensor quantization requires scalar zero_point, "
                f"got size {zero_point.size}"
            )
    else:
        # Per-channel quantization
        if scale.size != output_size:
            raise ValueError(
                f"Per-channel quantization requires scale.size == output_size "
                f"({output_size}), got {scale.size}"
            )
        if zero_point.size != output_size:
            raise ValueError(
                f"Per-channel quantization requires zero_point.size == output_size "
                f"({output_size}), got {zero_point.size}"
            )


def try_get_dequantized_weight(tensor_name: str, analyzer) -> Optional[np.ndarray]:
    """
    Try to resolve a dequantized weight tensor.

    Returns dequantized numpy array or None.
    """
    # Find producer layer
    producer_layer = None
    for layer in analyzer.layers:
        if tensor_name in layer["outputs"]:
            producer_layer = layer
            break

    if producer_layer is None or producer_layer["op_type"] != "DequantizeLinear":
        return None

    inputs = producer_layer["inputs"]
    if len(inputs) < 3:
        return None

    quantized_weight = analyzer.weights.get(inputs[0])
    scale = analyzer.weights.get(inputs[1])
    zero_point = analyzer.weights.get(inputs[2])

    if quantized_weight is None or scale is None or zero_point is None:
        return None

    # Dequantize
    try:
        if scale.ndim == 0:
            dequantized = scale * (
                quantized_weight.astype(np.float32) - zero_point.astype(np.float32)
            )
        else:
            axis = producer_layer.get("attributes", {}).get("axis", 1)
            shape = [1] * quantized_weight.ndim
            shape[axis] = -1
            scale_reshaped = scale.reshape(shape)
            zero_point_reshaped = zero_point.reshape(shape)
            dequantized = scale_reshaped * (
                quantized_weight.astype(np.float32)
                - zero_point_reshaped.astype(np.float32)
            )
        return dequantized
    except Exception as e:
        logger.error(f"Failed to dequantize {tensor_name}: {e}")
        return None
