"""
IR to Structured Text Code Generation Module

This module is responsible for generating Structured Text (ST) code from the intermediate representation (IR) of a neural network.
"""

from .types import *
from .st_code import STCode
import numpy as np


def generate_header(fb_name: str) -> STCode:
    """Generate function block header."""
    return STCode.from_lines(f"FUNCTION_BLOCK {fb_name}", "")


def generate_footer() -> STCode:
    """Generate function block footer."""
    return STCode.from_lines("END_FUNCTION_BLOCK")


def generate_var_input(network: NetworkIR) -> STCode:
    """Generate VAR_INPUT section."""
    return STCode.from_lines(
        "VAR_INPUT",
        f"    input_data : ARRAY[0..{network.input_size-1}] OF REAL;",
        "END_VAR",
        "",
    )


def generate_var_output(network: NetworkIR) -> STCode:
    """Generate VAR_OUTPUT section."""
    return STCode.from_lines(
        "VAR_OUTPUT",
        f"    output_data : ARRAY[0..{network.output_size-1}] OF REAL;",
        "END_VAR",
        "",
    )


def generate_reshape_code(
    layer: ReshapeLayer, input_var: str, output_var: str
) -> STCode:
    """Generate code corresponding to a ONNX Reshape layer."""
    same_size = layer.input_size == layer.output_size

    if same_size:
        # Copy input to output
        return STCode.from_lines(
            f"(* Layer {layer.layer_id}: Reshape (copy input to output) *)",
            f"FOR i := 0 TO {layer.input_size-1} DO",
            f"    {output_var}[i] := {input_var}[i];",
            "END_FOR;",
        )

    raise NotImplementedError(
        "Reshape layer with different sizes is not implemented yet."
    )

def generate_layer_variables(layer) -> STCode:
    """Generate variable declarations for a single layer."""
    lines = [f"(* Layer {layer.layer_id} variables *)"]
    if (
        isinstance(layer, MatMulLayer)
        or isinstance(layer, GemmLayer)
        or isinstance(layer, FusedGemmLayer)
    ):
        lines.append(
            f"weights_{layer.layer_id} : ARRAY[0..{layer.input_size-1}, 0..{layer.output_size-1}] OF REAL;"
        )
        if getattr(layer, "bias", None) is not None:
            lines.append(
                f"bias_{layer.layer_id} : ARRAY[0..{layer.output_size-1}] OF REAL;"
            )
        lines.append(
            f"layer_{layer.layer_id}_output : ARRAY[0..{layer.output_size-1}] OF REAL;"
        )
    elif isinstance(layer, AddLayer):
        lines.append(
            f"bias_{layer.layer_id} : ARRAY[0..{layer.output_size-1}] OF REAL;"
        )
        lines.append(
            f"layer_{layer.layer_id}_output : ARRAY[0..{layer.output_size-1}] OF REAL;"
        )
    elif isinstance(layer, ActivationLayer):
        lines.append(
            f"layer_{layer.layer_id}_output : ARRAY[0..{layer.output_size-1}] OF REAL;"
        )
    elif isinstance(layer, ReshapeLayer):
        lines.append(
            f"layer_{layer.layer_id}_output : ARRAY[0..{layer.output_size-1}] OF REAL;"
        )
    # TODO: quantize/dequantize layers....
    elif isinstance(layer, QuantizeLinearLayer) or isinstance(layer, DequantizeLinearLayer):
        lines.append(
            f"layer_{layer.layer_id}_output : ARRAY[0..{layer.output_size-1}] OF REAL;"
        )
    # TODO: dropout layers, conv layers, etc.
    else:
        raise NotImplementedError(
            f"Layer type {type(layer)} is not supported.")
    return STCode.from_lines(*lines)


def generate_var_section(network: NetworkIR) -> STCode:
    """Generate VAR section with all internal variables."""
    layer_vars = STCode.empty()
    for layer in network.layers:
        layer_vars = layer_vars + \
            generate_layer_variables(layer) + STCode.blank_line()

    temp_vars_lines = [
        "(* Temporary computation variables *)",
        "i : INT;",
        "j : INT;",
        "sum : REAL;",
    ]

    # Check if any layer uses softmax activation
    has_softmax = any(
        getattr(layer, "activation", None) == ActivationType.SOFTMAX
        for layer in network.layers
    )
    if has_softmax:
        temp_vars_lines.extend(
            [
                "max_val : REAL;",
                "exp_sum : REAL;",
            ]
        )

    temp_vars_lines.append("initialized : BOOL := FALSE;")

    temp_vars = STCode.from_lines(*temp_vars_lines)

    header = STCode.from_lines("VAR")
    footer = STCode.from_lines("END_VAR", "")

    return header + (layer_vars + temp_vars).indent() + footer


def format_1d_array_init(array: np.ndarray, var_name: str) -> STCode:
    """Format 1D array initialization."""
    lines = [f"{var_name}[{i}] := {array[i]:.6f};" for i in range(len(array))]
    return STCode.from_lines(*lines)


def format_2d_array_init(array: np.ndarray, var_name: str) -> STCode:
    """Format 2D array initialization."""
    lines = []
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            lines.append(f"{var_name}[{i},{j}] := {array[i, j]:.6f};")
    return STCode.from_lines(*lines)


def generate_layer_weight_init(layer) -> STCode:
    """Generate weight initialization for a single layer."""
    if hasattr(layer, "weights"):
        weight_init = format_2d_array_init(
            layer.weights, f"weights_{layer.layer_id}")
    else:
        weight_init = STCode.empty()
    if getattr(layer, "bias", None) is not None:
        bias_init = format_1d_array_init(layer.bias, f"bias_{layer.layer_id}")
        return weight_init + bias_init
    else:
        return weight_init


def generate_weight_initialization(network: NetworkIR) -> STCode:
    """Generate complete weight initialization block."""
    init_code = STCode.empty()

    for layer in network.layers:
        init_code = init_code + \
            generate_layer_weight_init(layer) + STCode.blank_line()

    header = STCode.from_lines(
        "(* Initialize weights (one-time setup) *)", "IF NOT initialized THEN"
    )
    footer = STCode.from_lines("    initialized := TRUE;", "END_IF;", "")

    return header + init_code.indent() + footer


def generate_activation_layer_code(
    layer: ActivationLayer, input_var: str, output_var: str
) -> STCode:
    comment = f"(* Layer {layer.layer_id}: Activation ({layer.activation.name}) *)"
    lines = [comment] + generate_activation_code(
        layer.activation, input_var, output_var, layer.output_size
    )
    return STCode.from_lines(*lines)


def generate_activation_code(
    activation: ActivationType, input_var: str, output_var: str, size: int
) -> list[str]:
    """Return activation code lines (without comment)."""
    match activation:
        case ActivationType.RELU:
            return [
                f"FOR i := 0 TO {size-1} DO",
                f"    {output_var}[i] := MAX({input_var}[i], 0.0);",
                "END_FOR;",
            ]
        case ActivationType.SIGMOID:
            return [
                f"FOR i := 0 TO {size-1} DO",
                f"    {output_var}[i] := 1.0 / (1.0 + EXP(-{input_var}[i]));",
                "END_FOR;",
            ]
        case ActivationType.TANH:
            return [
                f"FOR i := 0 TO {size-1} DO",
                f"    {output_var}[i] := (EXP({input_var}[i]) - EXP(-{input_var}[i])) / (EXP({input_var}[i]) + EXP(-{input_var}[i]));",
                "END_FOR;",
            ]
        case ActivationType.SOFTMAX:
            return [
                f"max_val := {input_var}[0];",
                f"FOR i := 1 TO {size-1} DO",
                f"    IF {input_var}[i] > max_val THEN",
                f"        max_val := {input_var}[i];",
                "    END_IF;",
                "END_FOR;",
                "",
                "exp_sum := 0.0;",
                f"FOR i := 0 TO {size-1} DO",
                f"    {output_var}[i] := EXP({input_var}[i] - max_val);",
                f"    exp_sum := exp_sum + {output_var}[i];",
                "END_FOR;",
                "",
                f"FOR i := 0 TO {size-1} DO",
                f"    {output_var}[i] := {output_var}[i] / exp_sum;",
                "END_FOR;",
            ]
        case _:
            return []


def generate_matmul_code(layer: MatMulLayer, input_var: str, output_var: str) -> STCode:
    # MatMul: Y = A * B
    return STCode.from_lines(
        f"(* Layer {layer.layer_id}: MatMul *)",
        f"FOR j := 0 TO {layer.output_size-1} DO",
        "    sum := 0.0;",
        f"    FOR i := 0 TO {layer.input_size-1} DO",
        f"        sum := sum + {input_var}[i] * weights_{layer.layer_id}[i,j];",
        "    END_FOR;",
        f"    {output_var}[j] := sum;",
        "END_FOR;",
    )


def generate_gemm_code(layer: GemmLayer, input_var: str, output_var: str) -> STCode:
    # Gemm: Y = alpha * A * B + beta * C
    bias_line = (
        f"{output_var}[j] := {layer.alpha} * sum + {layer.beta} * bias_{layer.layer_id}[j];"
        if layer.bias is not None
        else f"{output_var}[j] := {layer.alpha} * sum;"
    )
    return STCode.from_lines(
        f"(* Layer {layer.layer_id}: Gemm *)",
        f"FOR j := 0 TO {layer.output_size-1} DO",
        "    sum := 0.0;",
        f"    FOR i := 0 TO {layer.input_size-1} DO",
        f"        sum := sum + {input_var}[i] * weights_{layer.layer_id}[i,j];",
        "    END_FOR;",
        f"    {bias_line}",
        "END_FOR;",
    )


def generate_fused_gemm_code(
    layer: FusedGemmLayer, input_var: str, output_var: str
) -> STCode:
    # FusedGemm: Gemm followed by Activation

    # TODO: This should be the optimized version instead of just calling both sequentially
    gemm_code = generate_gemm_code(layer, input_var, output_var)
    activation_code = generate_activation_code(
        layer.activation, output_var, layer.output_size
    )
    return gemm_code + activation_code


def generate_add_code(layer: AddLayer, input_var: str, output_var: str) -> STCode:
    return STCode.from_lines(
        f"(* Layer {layer.layer_id}: Add (Bias) *)",
        f"FOR i := 0 TO {layer.output_size-1} DO",
        f"    {output_var}[i] := {input_var}[i] + bias_{layer.layer_id}[i];",
        "END_FOR;",
    )


def generate_forward_pass(network: NetworkIR) -> STCode:
    """Generate complete forward pass computation."""
    header = STCode.from_lines("(* Forward pass computation *)")
    forward_code = STCode.empty()
    current_input = "input_data"

    for idx, layer in enumerate(network.layers):
        is_last = idx == len(network.layers) - 1
        output_var = "output_data" if is_last else f"layer_{layer.layer_id}_output"

        if isinstance(layer, MatMulLayer):
            layer_code = generate_matmul_code(layer, current_input, output_var)
        elif isinstance(layer, AddLayer):
            layer_code = generate_add_code(layer, current_input, output_var)
        elif isinstance(layer, GemmLayer):
            layer_code = generate_gemm_code(layer, current_input, output_var)
        elif isinstance(layer, ReshapeLayer):
            layer_code = generate_reshape_code(
                layer, current_input, output_var)
        elif isinstance(layer, FusedGemmLayer):
            layer_code = generate_fused_gemm_code(
                layer, current_input, output_var)
        elif isinstance(layer, ActivationLayer):
            layer_code = generate_activation_layer_code(
                layer, current_input, output_var
            )
        else:
            layer_code = STCode.from_lines(
                f"(* Layer {layer.layer_id}: Unsupported layer type *)"
            )

        forward_code = forward_code + layer_code + STCode.blank_line()
        current_input = output_var

    return header + forward_code


def generate_function_block(
    network: NetworkIR, fb_name: str = "NeuralNetwork"
) -> STCode:
    """
    Generate complete function block from network IR.

    This is the main composition function that combines all parts.
    """
    return (
        generate_header(fb_name)
        + generate_var_input(network)
        + generate_var_output(network)
        + generate_var_section(network)
        + generate_weight_initialization(network)
        + generate_forward_pass(network)
        + generate_footer()
    )
