"""
IR to Structured Text Code Generation Module

This module is responsible for generating Structured Text (ST) code from the intermediate representation (IR) of a neural network.
"""

from ..types import *
from .st_code import *

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Type Utilities
# ============================================================================


def plc_type_from_dtype(dtype: str) -> str:
    """Map ONNX data type strings to PLC data types."""
    if dtype is None:
        raise ValueError("IR layer data type is None; tensor_info is incomplete.")

    match dtype:
        case "TensorProto.FLOAT":
            return "REAL"
        case "TensorProto.DOUBLE":
            return "LREAL"
        case "TensorProto.INT32":
            return "DINT"
        case "TensorProto.INT64":
            return "LINT"
        case "TensorProto.INT8":
            return "SINT"
        case "TensorProto.UINT8":
            return "USINT"
        case "TensorProto.INT16":
            return "INT"
        case "TensorProto.UINT16":
            return "UINT"
        case _:
            logging.warning(
                f"Data type {dtype} is not explicitly supported, adding placeholder."
            )
            raise NotImplementedError(f"Data type {dtype} is not supported.")


def plc_type_from_numpy_dtype(np_dtype: np.dtype) -> str:
    """Map NumPy dtype to PLC type."""
    if np_dtype == np.float32:
        return "REAL"
    elif np_dtype == np.float64:
        return "LREAL"
    elif np_dtype == np.int8:
        return "SINT"
    elif np_dtype == np.uint8:
        return "USINT"
    elif np_dtype == np.int16:
        return "INT"
    elif np_dtype == np.uint16:
        return "UINT"
    elif np_dtype == np.int32:
        return "DINT"
    elif np_dtype == np.int64:
        return "LINT"
    else:
        raise NotImplementedError(f"NumPy dtype {np_dtype} is not supported.")


def cast_function_for_dtype(np_dtype: np.dtype) -> str:
    """Get the PLC cast function for converting to REAL."""
    if np_dtype == np.int8:
        return "SINT_TO_REAL"
    elif np_dtype == np.uint8:
        return "USINT_TO_REAL"
    elif np_dtype == np.int16:
        return "INT_TO_REAL"
    elif np_dtype == np.uint16:
        return "UINT_TO_REAL"
    elif np_dtype == np.int32:
        return "DINT_TO_REAL"
    else:
        raise NotImplementedError(f"No cast function for dtype {np_dtype}")


# ============================================================================
# Header/Footer Generation
# ============================================================================


def generate_header(fb_name: str) -> STCode:
    """Generate function block header."""
    return STCode.from_lines(f"FUNCTION_BLOCK {fb_name}", "")


def generate_footer() -> STCode:
    """Generate function block footer."""
    return STCode.from_lines("END_FUNCTION_BLOCK", "")


def generate_input_output_vars(network: NetworkIR) -> STCode:
    """Generate VAR_INPUT and VAR_OUTPUT sections."""
    code = STCode.empty()

    first_layer_name = network.execution_order[0]
    first_layer = network.layers[first_layer_name]

    last_layer_name = network.execution_order[-1]
    last_layer = network.layers[last_layer_name]

    input_type = plc_type_from_dtype(first_layer.input_type)
    code += STCode.from_lines(
        "VAR_INPUT",
        f"    input_data : ARRAY[0..{first_layer.input_size - 1}] OF {input_type};",
        "END_VAR",
        "",
    )

    output_type = plc_type_from_dtype(last_layer.output_type)
    code += STCode.from_lines(
        "VAR_OUTPUT",
        f"    output_data : ARRAY[0..{last_layer.output_size - 1}] OF {output_type};",
        "END_VAR",
        "",
    )

    return code


# ============================================================================
# Constants Section - Unified Weight/Bias Generation
# ============================================================================


def generate_array_constant(
    name: str, values: np.ndarray, plc_type: str, is_integer: bool = False
) -> STCode:
    """
    Generate a constant array declaration.

    Args:
        name: Variable name
        values: NumPy array of values
        plc_type: PLC type string
        is_integer: If True, format as integers; otherwise as floats
    """
    flat_values = values.flatten()

    if is_integer:
        value_str = ", ".join(str(int(val)) for val in flat_values)
    else:
        value_str = ", ".join(f"{val:.6f}" for val in flat_values)

    return STCode.from_lines(
        f"{name} : ARRAY[0..{values.size - 1}] OF {plc_type} := [{value_str}];"
    )


def generate_scalar_constant(
    name: str, value: float | int, plc_type: str, is_integer: bool = False
) -> STCode:
    """Generate a scalar constant declaration."""
    if is_integer:
        value_str = str(int(value))
    else:
        value_str = str(float(value))

    return STCode.from_lines(f"{name} : {plc_type} := {value_str};")


def generate_layer_weights(layer) -> STCode:
    """
    Generate weight constants for a layer (handles both float and quantized).

    Returns all weight-related constants:
    - weights array
    - weight_scale (if quantized)
    - weight_zero_point (if quantized)
    """
    builder = STCodeBuilder()

    # Generate weights array
    is_quantized = hasattr(layer, "weight_scale") and layer.weight_scale is not None

    if is_quantized:
        # Quantized weights - store as integers
        weight_type = plc_type_from_numpy_dtype(layer.weights.dtype)
        weights_code = generate_array_constant(
            f"weights_{layer.layer_id}", layer.weights, weight_type, is_integer=True
        )
    else:
        # Float weights
        weight_type = plc_type_from_dtype(layer.input_type)
        weights_code = generate_array_constant(
            f"weights_{layer.layer_id}", layer.weights, weight_type, is_integer=False
        )

    builder.add_code(weights_code)

    # Generate quantization parameters if present
    if is_quantized:
        # Scale (always REAL)
        if layer.weight_scale.size == 1:
            builder.add_code(
                generate_scalar_constant(
                    f"weight_scale_{layer.layer_id}", layer.weight_scale.flat[0], "REAL"
                )
            )
        else:
            builder.add_code(
                generate_array_constant(
                    f"weight_scale_{layer.layer_id}", layer.weight_scale, "REAL"
                )
            )

        # Zero point (matches weight type)
        zp_type = plc_type_from_numpy_dtype(layer.weights.dtype)
        if layer.weight_zero_point.size == 1:
            builder.add_code(
                generate_scalar_constant(
                    f"weight_zero_point_{layer.layer_id}",
                    layer.weight_zero_point.flat[0],
                    zp_type,
                    is_integer=True,
                )
            )
        else:
            builder.add_code(
                generate_array_constant(
                    f"weight_zero_point_{layer.layer_id}",
                    layer.weight_zero_point,
                    zp_type,
                    is_integer=True,
                )
            )

    return builder.build()


def generate_layer_bias(layer) -> STCode:
    """Generate bias constant for a layer."""
    bias_type = plc_type_from_dtype(layer.output_type)
    return generate_array_constant(f"bias_{layer.layer_id}", layer.bias, bias_type)


def generate_layer_quantization_params(layer) -> STCode:
    """
    Generate quantization parameters for QuantizeLinear/DequantizeLinear layers.
    Only generates arrays for per-channel quantization (per-tensor is inlined).
    """
    # Only generate if per-channel (size > 1)
    if layer.scale.size == 1:
        return STCode.empty()

    builder = STCodeBuilder()

    # Scale array
    builder.add_code(
        generate_array_constant(f"scale_{layer.layer_id}", layer.scale, "REAL")
    )

    # Zero point array
    if isinstance(layer, QuantizeLinearLayer):
        dtype_str = layer.output_type
    else:  # DequantizeLinearLayer
        dtype_str = layer.input_type

    zp_type = plc_type_from_dtype(dtype_str)
    builder.add_code(
        generate_array_constant(
            f"zero_point_{layer.layer_id}", layer.zero_point, zp_type, is_integer=True
        )
    )

    return builder.build()


def generate_constants_section(network: NetworkIR) -> STCode:
    """Generate VAR CONSTANT section."""
    code = STCode.from_lines("VAR CONSTANT")

    for layer_name in network.execution_order:
        layer = network.layers[layer_name]
        has_constants = False

        # Weights (for linear layers)
        if hasattr(layer, "weights") and layer.weights is not None:
            code += generate_layer_weights(layer).indent()
            has_constants = True

        # Bias
        if hasattr(layer, "bias") and layer.bias is not None:
            code += generate_layer_bias(layer).indent()
            has_constants = True

        # Quantization parameters (for activation quantization only)
        if isinstance(layer, (QuantizeLinearLayer, DequantizeLinearLayer)):
            if layer.input_type is not None:  # Skip weight-only dequantization
                quant_params = generate_layer_quantization_params(layer)
                if quant_params.lines:
                    code += quant_params.indent()
                    has_constants = True

        if has_constants:
            code += STCode.blank_line()

    code += STCode.from_lines("END_VAR", "")
    return code


# ============================================================================
# VAR Section
# ============================================================================


def generate_var_section(network: NetworkIR) -> STCode:
    """Generate VAR section with all internal variables."""
    code = STCode.from_lines("VAR")

    # Layer output variables
    for layer_name in network.execution_order:
        layer = network.layers[layer_name]

        if any(network.is_network_output(out) for out in layer.outputs):
            continue

        plc_type = plc_type_from_dtype(layer.output_type)

        code += STCode.from_lines(
            f"(* Layer {layer.layer_id} output variable *)",
            f"layer_{layer.layer_id}_output : ARRAY[0..{layer.output_size-1}] OF {plc_type};",
        ).indent()

        code += STCode.blank_line()

    # Temporary computation variables
    code += STCode.from_lines(
        "    (* Temporary computation variables *)",
        "    i : DINT;",
        "    j : DINT;",
        "    sum : REAL;",
    )

    # Check if any layer uses softmax activation
    has_softmax = any(
        getattr(network.layers[layer_name], "activation", None)
        == ActivationType.SOFTMAX
        for layer_name in network.execution_order
    )

    if has_softmax:
        code += STCode.from_lines(
            "max_val : REAL;",
            "exp_sum : REAL;",
        ).indent()

    code += STCode.blank_line()
    code += STCode.from_lines("END_VAR", "")
    return code


# ============================================================================
# Layer Code Generators
# ============================================================================


def generate_activation_code(
    activation: ActivationType, input_var: str, output_var: str, size: int
) -> STCode:
    """Generate activation code using builder pattern."""
    builder = STCodeBuilder()

    if activation == ActivationType.RELU:
        builder.add_line(f"FOR i := 0 TO {size-1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := MAX({input_var}[i], 0.0);")
        builder.add_line("END_FOR;")

    elif activation == ActivationType.SIGMOID:
        builder.add_line(f"FOR i := 0 TO {size-1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := 1.0 / (1.0 + EXP(-{input_var}[i]));")
        builder.add_line("END_FOR;")

    elif activation == ActivationType.TANH:
        builder.add_line(f"FOR i := 0 TO {size-1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := (EXP({input_var}[i]) - EXP(-{input_var}[i])) / "
                f"(EXP({input_var}[i]) + EXP(-{input_var}[i]));"
            )
        builder.add_line("END_FOR;")

    elif activation == ActivationType.SOFTMAX:
        # Find maximum value
        builder.add_line(f"max_val := {input_var}[0];")
        builder.add_line(f"FOR i := 1 TO {size-1} DO")
        with builder.indent():
            builder.add_line(f"IF {input_var}[i] > max_val THEN")
            with builder.indent():
                builder.add_line(f"max_val := {input_var}[i];")
            builder.add_line("END_IF;")
        builder.add_line("END_FOR;")
        builder.add_line()

        # Compute exp sum
        builder.add_line("exp_sum := 0.0;")
        builder.add_line(f"FOR i := 0 TO {size-1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := EXP({input_var}[i] - max_val);")
            builder.add_line(f"exp_sum := exp_sum + {output_var}[i];")
        builder.add_line("END_FOR;")
        builder.add_line()

        # Normalize
        builder.add_line(f"FOR i := 0 TO {size-1} DO")
        with builder.indent():
            builder.add_line(f"{output_var}[i] := {output_var}[i] / exp_sum;")
        builder.add_line("END_FOR;")

    return builder.build()


def generate_activation_layer_code(
    layer: ActivationLayer, input_var: str, output_var: str
) -> STCode:
    """Generate activation layer code with comment."""
    code = STCode.from_lines(
        f"(* Layer {layer.layer_id}: Activation ({layer.activation.name}) *)"
    )
    code += generate_activation_code(
        layer.activation, input_var, output_var, layer.output_size
    )
    return code


def generate_matmul_code(layer: MatMulLayer, input_var: str, output_var: str) -> STCode:
    """Generate MatMul code - handles both float and quantized weights."""
    builder = STCodeBuilder()

    builder.add_line(f"(* Layer {layer.layer_id}: MatMul *)")

    is_quantized = hasattr(layer, "weight_scale") and layer.weight_scale is not None

    if is_quantized:
        # Quantized weights
        is_per_tensor = layer.weight_scale.size == 1
        cast_func = cast_function_for_dtype(layer.weights.dtype)

        builder.add_line(f"FOR j := 0 TO {layer.output_size-1} DO")
        with builder.indent():
            builder.add_line("sum := 0.0;")
            builder.add_line(f"FOR i := 0 TO {layer.input_size-1} DO")
            with builder.indent():
                if is_per_tensor:
                    builder.add_line(
                        f"sum := sum + {input_var}[i] * "
                        f"(weight_scale_{layer.layer_id} * {cast_func}(weights_{layer.layer_id}[i * {layer.output_size} + j] - weight_zero_point_{layer.layer_id}));"
                    )
                else:
                    builder.add_line(
                        f"sum := sum + {input_var}[i] * "
                        f"(weight_scale_{layer.layer_id}[j] * {cast_func}(weights_{layer.layer_id}[i * {layer.output_size} + j] - weight_zero_point_{layer.layer_id}[j]));"
                    )
            builder.add_line("END_FOR;")
            builder.add_line(f"{output_var}[j] := sum;")
        builder.add_line("END_FOR;")
    else:
        # Float weights
        builder.add_line(f"FOR j := 0 TO {layer.output_size-1} DO")
        with builder.indent():
            builder.add_line("sum := 0.0;")
            builder.add_line(f"FOR i := 0 TO {layer.input_size-1} DO")
            with builder.indent():
                builder.add_line(
                    f"sum := sum + {input_var}[i] * weights_{layer.layer_id}[i * {layer.output_size} + j];"
                )
            builder.add_line("END_FOR;")
            builder.add_line(f"{output_var}[j] := sum;")
        builder.add_line("END_FOR;")

    return builder.build()


def generate_gemm_code(layer: GemmLayer, input_var: str, output_var: str) -> STCode:
    """Generate Gemm code - handles both float and quantized weights."""
    builder = STCodeBuilder()

    builder.add_line(f"(* Layer {layer.layer_id}: Gemm *)")

    is_quantized = hasattr(layer, "weight_scale") and layer.weight_scale is not None

    if is_quantized:
        # Quantized weights
        is_per_tensor = layer.weight_scale.size == 1
        cast_func = cast_function_for_dtype(layer.weights.dtype)

        builder.add_line(f"FOR j := 0 TO {layer.output_size-1} DO")
        with builder.indent():
            builder.add_line("sum := 0.0;")
            builder.add_line(f"FOR i := 0 TO {layer.input_size-1} DO")
            with builder.indent():
                if is_per_tensor:
                    builder.add_line(
                        f"sum := sum + {input_var}[i] * "
                        f"(weight_scale_{layer.layer_id} * {cast_func}(weights_{layer.layer_id}[i * {layer.output_size} + j] - weight_zero_point_{layer.layer_id}));"
                    )
                else:
                    builder.add_line(
                        f"sum := sum + {input_var}[i] * "
                        f"(weight_scale_{layer.layer_id}[j] * {cast_func}(weights_{layer.layer_id}[i * {layer.output_size} + j] - weight_zero_point_{layer.layer_id}[j]));"
                    )
            builder.add_line("END_FOR;")

            if layer.bias is not None:
                builder.add_line(
                    f"{output_var}[j] := {layer.alpha} * sum + {layer.beta} * bias_{layer.layer_id}[j];"
                )
            else:
                builder.add_line(f"{output_var}[j] := {layer.alpha} * sum;")
        builder.add_line("END_FOR;")
    else:
        # Float weights
        builder.add_line(f"FOR j := 0 TO {layer.output_size-1} DO")
        with builder.indent():
            builder.add_line("sum := 0.0;")
            builder.add_line(f"FOR i := 0 TO {layer.input_size-1} DO")
            with builder.indent():
                builder.add_line(
                    f"sum := sum + {input_var}[i] * weights_{layer.layer_id}[i * {layer.output_size} + j];"
                )
            builder.add_line("END_FOR;")

            if layer.bias is not None:
                builder.add_line(
                    f"{output_var}[j] := {layer.alpha} * sum + {layer.beta} * bias_{layer.layer_id}[j];"
                )
            else:
                builder.add_line(f"{output_var}[j] := {layer.alpha} * sum;")
        builder.add_line("END_FOR;")

    return builder.build()


def generate_fused_gemm_code(
    layer: FusedGemmLayer, input_var: str, output_var: str
) -> STCode:
    """Generate FusedGemm layer code: Gemm followed by Activation"""
    code = generate_gemm_code(layer, input_var, output_var)
    code += STCode.blank_line()
    code += generate_activation_code(
        layer.activation, output_var, output_var, layer.output_size
    )
    return code


def generate_add_code(layer: AddLayer, input_var: str, output_var: str) -> STCode:
    """Generate Add (bias) layer code."""
    builder = STCodeBuilder()

    builder.add_line(f"(* Layer {layer.layer_id}: Add (Bias) *)")
    builder.add_line(f"FOR i := 0 TO {layer.output_size-1} DO")
    with builder.indent():
        builder.add_line(
            f"{output_var}[i] := {input_var}[i] + bias_{layer.layer_id}[i];"
        )
    builder.add_line("END_FOR;")

    return builder.build()


def generate_reshape_code(
    layer: ReshapeLayer, input_var: str, output_var: str
) -> STCode:
    """Generate code corresponding to a ONNX Reshape layer."""
    if layer.input_size != layer.output_size:
        raise NotImplementedError(
            "Reshape layer with different sizes is not implemented yet."
        )

    builder = STCodeBuilder()
    builder.add_line(f"(* Layer {layer.layer_id}: Reshape (copy input to output) *)")
    builder.add_line(f"FOR i := 0 TO {layer.input_size-1} DO")
    with builder.indent():
        builder.add_line(f"{output_var}[i] := {input_var}[i];")
    builder.add_line("END_FOR;")

    return builder.build()


def generate_quantize_linear_code(
    layer: QuantizeLinearLayer, input_var: str, output_var: str
) -> STCode:
    """Generate QuantizeLinear code: quantized = clip(round(input/scale) + zero_point)"""
    builder = STCodeBuilder()

    builder.add_line(f"(* Layer {layer.layer_id}: {layer.name} - QuantizeLinear *)")

    # Get bounds and cast function from output type
    output_plc_type = plc_type_from_dtype(layer.output_type)

    if layer.output_type == "TensorProto.INT8":
        min_val, max_val = -128, 127
        cast_func = "REAL_TO_SINT"
    elif layer.output_type == "TensorProto.UINT8":
        min_val, max_val = 0, 255
        cast_func = "REAL_TO_USINT"
    elif layer.output_type == "TensorProto.INT16":
        min_val, max_val = -32768, 32767
        cast_func = "REAL_TO_INT"
    elif layer.output_type == "TensorProto.UINT16":
        min_val, max_val = 0, 65535
        cast_func = "REAL_TO_UINT"
    else:
        min_val, max_val = -2147483648, 2147483647
        cast_func = "REAL_TO_DINT"

    is_per_tensor = layer.scale.size == 1

    if is_per_tensor:
        # Per-tensor quantization
        scale_val = layer.scale.flat[0]
        zero_point_val = layer.zero_point.flat[0]

        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := LIMIT({min_val}, "
                f"{cast_func}(ROUND({input_var}[i] / {scale_val}) + {zero_point_val}), "
                f"{max_val});"
            )
        builder.add_line("END_FOR;")
    else:
        # Per-channel quantization
        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := LIMIT({min_val}, "
                f"{cast_func}(ROUND({input_var}[i] / scale_{layer.layer_id}[i]) + zero_point_{layer.layer_id}[i]), "
                f"{max_val});"
            )
        builder.add_line("END_FOR;")

    return builder.build()


def generate_dequantize_linear_code(
    layer: DequantizeLinearLayer, input_var: str, output_var: str
) -> STCode:
    """Generate DequantizeLinear code: float = scale * (quantized - zero_point)"""
    builder = STCodeBuilder()

    builder.add_line(f"(* Layer {layer.layer_id}: {layer.name} - DequantizeLinear *)")

    # Get cast function from input type
    if layer.input_type == "TensorProto.INT8":
        cast_func = "SINT_TO_REAL"
    elif layer.input_type == "TensorProto.UINT8":
        cast_func = "USINT_TO_REAL"
    elif layer.input_type == "TensorProto.INT16":
        cast_func = "INT_TO_REAL"
    elif layer.input_type == "TensorProto.UINT16":
        cast_func = "UINT_TO_REAL"
    else:
        cast_func = "DINT_TO_REAL"

    is_per_tensor = layer.scale.size == 1

    if is_per_tensor:
        # Per-tensor dequantization
        scale_val = layer.scale.flat[0]
        zero_point_val = layer.zero_point.flat[0]

        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := {scale_val} * "
                f"{cast_func}({input_var}[i] - {zero_point_val});"
            )
        builder.add_line("END_FOR;")
    else:
        # Per-channel dequantization
        builder.add_line(f"FOR i := 0 TO {layer.output_size - 1} DO")
        with builder.indent():
            builder.add_line(
                f"{output_var}[i] := scale_{layer.layer_id}[i] * "
                f"{cast_func}({input_var}[i] - zero_point_{layer.layer_id}[i]);"
            )
        builder.add_line("END_FOR;")

    return builder.build()


# ============================================================================
# Forward Pass Generation
# ============================================================================

# Mapping from layer type to code generator
LAYER_CODE_GENERATORS = {
    MatMulLayer: generate_matmul_code,
    AddLayer: generate_add_code,
    GemmLayer: generate_gemm_code,
    FusedGemmLayer: generate_fused_gemm_code,
    ReshapeLayer: generate_reshape_code,
    ActivationLayer: generate_activation_layer_code,
    QuantizeLinearLayer: generate_quantize_linear_code,
    DequantizeLinearLayer: generate_dequantize_linear_code,
}


def generate_forward_pass(network: NetworkIR) -> STCode:
    """Generate the forward pass computation code for all layers."""
    code = STCode.empty()
    for layer_name in network.execution_order:
        layer = network.layers[layer_name]

        # Determine input variable name
        if len(layer.inputs) == 1:
            input_tensor = layer.inputs[0]
            if network.is_network_input(input_tensor):
                input_var = "input_data"
            else:
                producer_layer_name = network.tensor_producers[input_tensor]
                producer_layer = network.layers[producer_layer_name]
                input_var = f"layer_{producer_layer.layer_id}_output"
        else:
            # For layers with multiple inputs, use the first non-weight input
            input_var = None
            for inp_tensor in layer.inputs:
                if network.is_network_input(inp_tensor):
                    input_var = "input_data"
                    break
                elif inp_tensor in network.tensor_producers:
                    producer_layer_name = network.tensor_producers[inp_tensor]
                    producer_layer = network.layers[producer_layer_name]
                    input_var = f"layer_{producer_layer.layer_id}_output"
                    break

            if input_var is None:
                raise ValueError(
                    f"Cannot determine input variable for layer {layer_name}"
                )

        # Determine output variable name
        output_tensor = layer.outputs[0]
        if network.is_network_output(output_tensor):
            output_var = "output_data"
        else:
            output_var = f"layer_{layer.layer_id}_output"

        # Generate code for this layer
        layer_type = type(layer)
        if layer_type in LAYER_CODE_GENERATORS:
            layer_code = LAYER_CODE_GENERATORS[layer_type](layer, input_var, output_var)
            code += layer_code
            code += STCode.blank_line()
        else:
            logger.warning(f"No code generator for layer type {layer_type.__name__}")

    return code


# ============================================================================
# Main Entry Point
# ============================================================================


def generate_function_block(
    network: NetworkIR, fb_name: str = "NeuralNetwork"
) -> STCode:
    """Generate complete function block code."""
    logger.info(
        f"Generating function block '{fb_name}' with {len(network.layers)} layers"
    )

    code = STCode.empty()
    code += generate_header(fb_name)
    code += generate_input_output_vars(network)
    code += generate_constants_section(network)
    code += generate_var_section(network)
    code += generate_forward_pass(network)
    code += generate_footer()

    logger.info(f"Generated {len(code.lines)} lines of ST code.")
    return code


def translate_ir_to_st(
    network: NetworkIR, fb_name: str = "NeuralNetwork"
) -> str:
    """Translate the given NetworkIR to Structured Text code."""
    return str(generate_function_block(network, fb_name))

    # TODO: might need to add openplc config / straton config generation later