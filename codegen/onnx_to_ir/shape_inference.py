"""
Central shape inference for IR construction.
Infers shapes from operation semantics when ONNX shape info is incomplete.
I.e. infer output shape based on the operation (after tensors are resolved, during layer extraction).
"""

import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def infer_matmul_output_shape(
    input_shape: Tuple[int, ...], weight_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Infer output shape for MatMul operation.

    MatMul: (M, K) @ (K, N) -> (M, N)
    For batched: (..., M, K) @ (K, N) -> (..., M, N)

    For our PLC use case, we flatten batches, so:
    (K,) @ (K, N) -> (N,)
    """
    if not weight_shape or len(weight_shape) < 2:
        logger.warning(f"Invalid weight shape for MatMul: {weight_shape}")
        return ()

    # Weight is always 2D: (input_features, output_features)
    output_features = weight_shape[1]

    if not input_shape:
        return (output_features,)

    # For PLCs, we typically work with flattened 1D inputs
    # Keep batch dimensions if present, replace last dim with output_features
    if len(input_shape) > 1:
        return (*input_shape[:-1], output_features)
    else:
        return (output_features,)


def infer_gemm_output_shape(
    input_shape: Tuple[int, ...], weight_shape: Tuple[int, ...], transB: bool = False
) -> Tuple[int, ...]:
    """
    Infer output shape for Gemm operation.

    Gemm: Y = alpha * A @ B^T + beta * C  (if transB=True)
          Y = alpha * A @ B + beta * C     (if transB=False)

    Args:
        input_shape: Shape of input A
        weight_shape: Shape of weight B
        transB: Whether B is transposed
    """
    if not weight_shape or len(weight_shape) < 2:
        logger.warning(f"Invalid weight shape for Gemm: {weight_shape}")
        return ()

    # Determine output features based on transB
    if transB:
        # B is (output_features, input_features), transposed becomes (input_features, output_features)
        output_features = weight_shape[0]
    else:
        # B is (input_features, output_features)
        output_features = weight_shape[1]

    # Gemm typically produces 1D output (batch dimension removed or kept as 1)
    return (output_features,)


def infer_element_wise_output_shape(input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Infer output shape for element-wise operations (ReLU, Sigmoid, Tanh, etc.).

    Element-wise operations preserve input shape.
    """
    return input_shape


def infer_add_output_shape(
    input_shape: Tuple[int, ...], bias_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Infer output shape for Add operation.

    Add with broadcasting: typically input + bias where bias is 1D.
    """
    # Add preserves the larger shape (broadcasting rules)
    if not input_shape:
        return bias_shape
    if not bias_shape:
        return input_shape

    # In most cases, bias is 1D and broadcasts to input shape
    return input_shape


def infer_reshape_output_shape(
    input_shape: Tuple[int, ...], target_shape: Optional[Tuple[int, ...]]
) -> Tuple[int, ...]:
    """
    Infer output shape for Reshape operation.

    Args:
        input_shape: Input tensor shape
        target_shape: Target shape (may contain -1 for inferred dimension)

    Returns:
        Resolved output shape
    """
    if not target_shape:
        # No target shape provided - flatten to 1D
        if input_shape:
            total_size = int(np.prod(input_shape))
            return (total_size,)
        return ()

    # Handle -1 in target shape (infer dimension)
    if -1 in target_shape:
        input_size = int(np.prod(input_shape)) if input_shape else 0
        known_dims = [d for d in target_shape if d > 0]
        known_prod = int(np.prod(known_dims)) if known_dims else 1

        if known_prod == 0:
            logger.warning(f"Invalid target shape {target_shape}")
            return ()

        inferred_dim = input_size // known_prod
        resolved_shape = tuple(inferred_dim if d == -1 else d for d in target_shape)
        return resolved_shape

    # All dimensions are specified
    return target_shape


def infer_softmax_output_shape(input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Infer output shape for Softmax operation.

    Softmax preserves input shape.
    """
    return input_shape


def infer_quantize_output_shape(input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Infer output shape for QuantizeLinear operation.

    Quantization preserves shape, only changes dtype.
    """
    return input_shape


def infer_dequantize_output_shape(input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Infer output shape for DequantizeLinear operation.

    Dequantization preserves shape, only changes dtype.
    """
    return input_shape


def infer_layer_shapes(
    layer_dict: Dict[str, Any],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Infer input and output shapes for a layer based on operation type.

    This is the main entry point for shape inference. It tries to use ONNX
    tensor_info shapes first, and falls back to operation-specific inference.

    Args:
        layer_dict: Enriched layer dict with 'resolved_inputs' and 'resolved_outputs'

    Returns:
        (input_shape, output_shape) - tuples of integers only (no symbolic dims)
    """
    op_type = layer_dict["op_type"]
    resolved_inputs = layer_dict["resolved_inputs"]
    resolved_outputs = layer_dict["resolved_outputs"]

    # Get input shape (first data tensor, skip weights)
    data_input = None
    for inp in resolved_inputs:
        if not inp.is_weight:
            data_input = inp
            break

    if data_input is None:
        data_input = resolved_inputs[0]

    input_shape = data_input.shape if data_input.shape else ()

    # Try to get shape from ONNX tensor_info first
    output_tensor_info_shape = resolved_outputs[0].shape if resolved_outputs else ()

    # If output shape is valid (has at least one dimension), use it
    if output_tensor_info_shape:
        logger.debug(f"{op_type}: Using ONNX output shape {output_tensor_info_shape}")
        return input_shape, output_tensor_info_shape

    # Otherwise, infer from operation semantics
    logger.debug(f"{op_type}: Inferring output shape (ONNX shape empty)")

    if op_type == "MatMul":
        weight_tensor = resolved_inputs[1]
        output_shape = infer_matmul_output_shape(input_shape, weight_tensor.shape)

    elif op_type in ["Gemm", "FusedGemm"]:
        weight_tensor = resolved_inputs[1]
        attrs = layer_dict.get("attributes", {})
        transB = attrs.get("transB", 0) == 1
        output_shape = infer_gemm_output_shape(input_shape, weight_tensor.shape, transB)

    elif op_type in ["Relu", "Sigmoid", "Tanh"]:
        output_shape = infer_element_wise_output_shape(input_shape)

    elif op_type == "Softmax":
        output_shape = infer_softmax_output_shape(input_shape)

    elif op_type == "Add":
        # Check if second input is bias
        if len(resolved_inputs) > 1:
            bias_tensor = resolved_inputs[1]
            output_shape = infer_add_output_shape(input_shape, bias_tensor.shape)
        else:
            output_shape = input_shape

    elif op_type == "Reshape":
        # Try to get target shape from second input (shape tensor)
        target_shape = None
        if len(resolved_inputs) > 1 and resolved_inputs[1].is_weight:
            shape_array = resolved_inputs[1].value
            if shape_array is not None:
                # Filter out 0 and keep positive dimensions, convert -1
                target_shape = tuple(int(d) for d in shape_array if d != 0)
        output_shape = infer_reshape_output_shape(input_shape, target_shape)

    elif op_type == "QuantizeLinear":
        output_shape = infer_quantize_output_shape(input_shape)

    elif op_type == "DequantizeLinear":
        output_shape = infer_dequantize_output_shape(input_shape)

    else:
        logger.warning(f"No shape inference for op_type '{op_type}', using input shape")
        output_shape = input_shape

    logger.debug(f"{op_type}: Inferred {input_shape} -> {output_shape}")
    return input_shape, output_shape


def get_feature_sizes(
    input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]
) -> Tuple[int, int]:
    """
    Get input and output feature sizes (ignoring batch dimensions).

    For PLC code generation, we typically work with flattened 1D arrays,
    so we take the last dimension as the feature size, or the total size
    if the shape is 1D.

    Args:
        input_shape: Input tensor shape
        output_shape: Output tensor shape

    Returns:
        (input_size, output_size) - number of features/elements
    """
    # For 1D shapes, use the dimension directly
    # For multi-dimensional, use the last dimension (feature dimension)
    input_size = input_shape[-1] if input_shape else 0
    output_size = output_shape[-1] if output_shape else 0

    return input_size, output_size


def validate_inferred_shapes(
    layer_name: str,
    op_type: str,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    weight_shape: Optional[Tuple[int, ...]] = None,
) -> bool:
    """
    Validate that inferred shapes are consistent with operation semantics.

    Args:
        layer_name: Name of the layer (for logging)
        op_type: Operation type
        input_shape: Inferred input shape
        output_shape: Inferred output shape
        weight_shape: Weight shape (if applicable)

    Returns:
        True if shapes are valid, raises ValueError otherwise
    """
    if not output_shape:
        raise ValueError(f"Layer {layer_name} ({op_type}): Output shape is empty")

    if not input_shape:
        logger.warning(f"Layer {layer_name} ({op_type}): Input shape is empty")

    # Operation-specific validation
    if op_type in ["MatMul", "Gemm", "FusedGemm"] and weight_shape:
        if len(weight_shape) != 2:
            raise ValueError(
                f"Layer {layer_name} ({op_type}): "
                f"Weight must be 2D, got {weight_shape}"
            )

        # Check dimension compatibility
        if input_shape and weight_shape:
            input_features = input_shape[-1]
            weight_input_features = weight_shape[0]

            if input_features != weight_input_features:
                raise ValueError(
                    f"Layer {layer_name} ({op_type}): "
                    f"Dimension mismatch - input features {input_features} "
                    f"!= weight input features {weight_input_features}"
                )

    return True
