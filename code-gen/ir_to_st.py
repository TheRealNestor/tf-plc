"""
IR to Structured Text Code Generation Module

This module is responsible for generating Structured Text (ST) code from the intermediate representation (IR) of a neural network.
"""

from typing import Tuple
from .types import NetworkIR, DenseLayer, ActivationType
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


def generate_layer_variables(layer: DenseLayer) -> STCode:
    """Generate variable declarations for a single layer."""
    lines = [
        f"(* Layer {layer.layer_id} variables *)",
        f"weights_{layer.layer_id} : ARRAY[0..{layer.input_size-1}, 0..{layer.output_size-1}] OF REAL;",
    ]

    if layer.bias is not None:
        lines.append(
            f"bias_{layer.layer_id} : ARRAY[0..{layer.output_size-1}] OF REAL;"
        )

    lines.append(
        f"layer_{layer.layer_id}_output : ARRAY[0..{layer.output_size-1}] OF REAL;"
    )

    return STCode.from_lines(*lines)


def generate_var_section(network: NetworkIR) -> STCode:
    """Generate VAR section with all internal variables."""
    layer_vars = STCode.empty()
    for layer in network.layers:
        layer_vars = layer_vars + generate_layer_variables(layer) + STCode.blank_line()

    # Check if any layer uses softmax activation
    has_softmax = any(
        layer.activation == ActivationType.SOFTMAX for layer in network.layers
    )

    temp_vars_lines = [
        "(* Temporary computation variables *)",
        "i, j : INT;",
        "sum : REAL;",
    ]

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
            lines.append(f"{var_name}[{i},{j}] := {array[i,j]:.6f};")
    return STCode.from_lines(*lines)


def generate_layer_weight_init(layer: DenseLayer) -> STCode:
    """Generate weight initialization for a single layer."""
    comment = STCode.from_lines(f"(* Initialize layer {layer.layer_id} weights *)")

    weight_init = format_2d_array_init(layer.weights, f"weights_{layer.layer_id}")

    if layer.bias is not None:
        bias_init = format_1d_array_init(layer.bias, f"bias_{layer.layer_id}")
        return comment + weight_init + bias_init
    else:
        return comment + weight_init


def generate_weight_initialization(network: NetworkIR) -> STCode:
    """Generate complete weight initialization block."""
    init_code = STCode.empty()

    for layer in network.layers:
        init_code = init_code + generate_layer_weight_init(layer) + STCode.blank_line()

    header = STCode.from_lines(
        "(* Initialize weights (one-time setup) *)", "IF NOT initialized THEN"
    )
    footer = STCode.from_lines("    initialized := TRUE;", "END_IF;", "")

    return header + init_code.indent() + footer


def generate_activation_code(
    activation: ActivationType, array_name: str, size: int
) -> STCode:
    """Generate activation function code based on type."""
    match activation:
        case ActivationType.RELU:
            return STCode.from_lines(
                "(* ReLU activation *)",
                f"FOR i := 0 TO {size-1} DO",
                f"    IF {array_name}[i] > 0.0 THEN",
                f"        {array_name}[i] := {array_name}[i];",
                "    ELSE",
                f"        {array_name}[i] := 0.0;",
                "    END_IF;",
                "END_FOR;",
            )

        case ActivationType.SIGMOID:
            return STCode.from_lines(
                "(* Sigmoid activation *)",
                f"FOR i := 0 TO {size-1} DO",
                f"    {array_name}[i] := 1.0 / (1.0 + EXP(-{array_name}[i]));",
                "END_FOR;",
            )

        case ActivationType.TANH:
            return STCode.from_lines(
                "(* Tanh activation *)",
                f"FOR i := 0 TO {size-1} DO",
                f"    {array_name}[i] := (EXP({array_name}[i]) - EXP(-{array_name}[i])) / (EXP({array_name}[i]) + EXP(-{array_name}[i]));",
                "END_FOR;",
            )

        case ActivationType.SOFTMAX:
            return STCode.from_lines(
                "(* Softmax activation *)",
                f"max_val := {array_name}[0];",
                f"FOR i := 1 TO {size-1} DO",
                f"    IF {array_name}[i] > max_val THEN",
                f"        max_val := {array_name}[i];",
                "    END_IF;",
                "END_FOR;",
                "",
                "exp_sum := 0.0;",
                f"FOR i := 0 TO {size-1} DO",
                f"    {array_name}[i] := EXP({array_name}[i] - max_val);",
                f"    exp_sum := exp_sum + {array_name}[i];",
                "END_FOR;",
                "",
                f"FOR i := 0 TO {size-1} DO",
                f"    {array_name}[i] := {array_name}[i] / exp_sum;",
                "END_FOR;",
            )

        case _:
            return STCode.empty()


def generate_matmul_code(layer: DenseLayer, input_var: str, output_var: str) -> STCode:
    """Generate matrix multiplication code for a dense layer."""
    has_bias = layer.bias is not None

    bias_line = (
        f"{output_var}[j] := sum + bias_{layer.layer_id}[j];"
        if has_bias
        else f"{output_var}[j] := sum;"
    )

    return STCode.from_lines(
        f"(* Layer {layer.layer_id}: Dense (MatMul + Bias) *)",
        f"FOR j := 0 TO {layer.output_size-1} DO",
        "    sum := 0.0;",
        f"    FOR i := 0 TO {layer.input_size-1} DO",
        f"        sum := sum + {input_var}[i] * weights_{layer.layer_id}[i,j];",
        "    END_FOR;",
        f"    {bias_line}",
        "END_FOR;",
    )


def generate_layer_forward_pass(
    layer: DenseLayer, input_var: str, is_last: bool
) -> Tuple[STCode, str]:
    """
    Generate forward pass code for a single layer.

    Args:
        layer: Dense layer to generate code for
        input_var: Name of input variable
        is_last: Whether this is the last layer

    Returns:
        Tuple of (generated code, output variable name)
    """
    output_var = "output_data" if is_last else f"layer_{layer.layer_id}_output"

    matmul_code = generate_matmul_code(layer, input_var, output_var)
    activation_code = generate_activation_code(
        layer.activation, output_var, layer.output_size
    )

    layer_code = matmul_code
    if activation_code.lines:
        layer_code = layer_code + activation_code

    return (layer_code + STCode.blank_line(), output_var)


def generate_forward_pass(network: NetworkIR) -> STCode:
    """Generate complete forward pass computation."""
    header = STCode.from_lines("(* Forward pass computation *)")

    forward_code = STCode.empty()
    current_input = "input_data"

    for idx, layer in enumerate(network.layers):
        is_last = idx == len(network.layers) - 1
        layer_code, output_var = generate_layer_forward_pass(
            layer, current_input, is_last
        )
        forward_code = forward_code + layer_code
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

