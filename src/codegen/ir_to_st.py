"""
IR to Structured Text Code Generation Module

This module is responsible for generating Structured Text (ST) code from the intermediate representation (IR) of a neural network.
"""

from .types import *
from .st_code import *

import logging
logger = logging.getLogger(__name__)


# tensor_dtype: (numpy type, storage type, string name)
# The storage type is the type used to store the tensor in the *_data field of
# a TensorProto. All available fields are float_data, int32_data, int64_data,
# string_data, uint64_data and double_data.
# TENSOR_TYPE_MAP: dict[int, TensorDtypeMap] = {
#     int(TensorProto.FLOAT): TensorDtypeMap(
#         np.dtype("float32"), int(TensorProto.FLOAT), "TensorProto.FLOAT"
#     ),
#     int(TensorProto.UINT8): TensorDtypeMap(
#         np.dtype("uint8"), int(TensorProto.INT32), "TensorProto.UINT8"
#     ),
#     int(TensorProto.INT8): TensorDtypeMap(
#         np.dtype("int8"), int(TensorProto.INT32), "TensorProto.INT8"
#     ),
#     int(TensorProto.UINT16): TensorDtypeMap(
#         np.dtype("uint16"), int(TensorProto.INT32), "TensorProto.UINT16"
#     ),
#     int(TensorProto.INT16): TensorDtypeMap(
#         np.dtype("int16"), int(TensorProto.INT32), "TensorProto.INT16"
#     ),
#     int(TensorProto.INT32): TensorDtypeMap(
#         np.dtype("int32"), int(TensorProto.INT32), "TensorProto.INT32"
#     ),
#     int(TensorProto.INT64): TensorDtypeMap(
#         np.dtype("int64"), int(TensorProto.INT64), "TensorProto.INT64"
#     ),
#     int(TensorProto.BOOL): TensorDtypeMap(
#         np.dtype("bool"), int(TensorProto.INT32), "TensorProto.BOOL"
#     ),
#     int(TensorProto.FLOAT16): TensorDtypeMap(
#         np.dtype("float16"), int(TensorProto.INT32), "TensorProto.FLOAT16"
#     ),
#     int(TensorProto.BFLOAT16): TensorDtypeMap(
#         np.dtype(ml_dtypes.bfloat16),
#         int(TensorProto.INT32),
#         "TensorProto.BFLOAT16",
#     ),
#     int(TensorProto.DOUBLE): TensorDtypeMap(
#         np.dtype("float64"), int(TensorProto.DOUBLE), "TensorProto.DOUBLE"
#     ),
#     int(TensorProto.COMPLEX64): TensorDtypeMap(
#         np.dtype("complex64"), int(TensorProto.FLOAT), "TensorProto.COMPLEX64"
#     ),
#     int(TensorProto.COMPLEX128): TensorDtypeMap(
#         np.dtype("complex128"),
#         int(TensorProto.DOUBLE),
#         "TensorProto.COMPLEX128",
#     ),
#     int(TensorProto.UINT32): TensorDtypeMap(
#         np.dtype("uint32"), int(TensorProto.UINT64), "TensorProto.UINT32"
#     ),
#     int(TensorProto.UINT64): TensorDtypeMap(
#         np.dtype("uint64"), int(TensorProto.UINT64), "TensorProto.UINT64"
#     ),
#     int(TensorProto.STRING): TensorDtypeMap(
#         np.dtype("object"), int(TensorProto.STRING), "TensorProto.STRING"
#     ),
#     int(TensorProto.FLOAT8E4M3FN): TensorDtypeMap(
#         np.dtype(ml_dtypes.float8_e4m3fn),
#         int(TensorProto.INT32),
#         "TensorProto.FLOAT8E4M3FN",
#     ),
#     int(TensorProto.FLOAT8E4M3FNUZ): TensorDtypeMap(
#         np.dtype(ml_dtypes.float8_e4m3fnuz),
#         int(TensorProto.INT32),
#         "TensorProto.FLOAT8E4M3FNUZ",
#     ),
#     int(TensorProto.FLOAT8E5M2): TensorDtypeMap(
#         np.dtype(ml_dtypes.float8_e5m2),
#         int(TensorProto.INT32),
#         "TensorProto.FLOAT8E5M2",
#     ),
#     int(TensorProto.FLOAT8E5M2FNUZ): TensorDtypeMap(
#         np.dtype(ml_dtypes.float8_e5m2fnuz),
#         int(TensorProto.INT32),
#         "TensorProto.FLOAT8E5M2FNUZ",
#     ),
#     int(TensorProto.UINT4): TensorDtypeMap(
#         np.dtype(ml_dtypes.uint4), int(TensorProto.INT32), "TensorProto.UINT4"
#     ),
#     int(TensorProto.INT4): TensorDtypeMap(
#         np.dtype(ml_dtypes.int4), int(TensorProto.INT32), "TensorProto.INT4"
#     ),
#     int(TensorProto.FLOAT4E2M1): TensorDtypeMap(
#         np.dtype(ml_dtypes.float4_e2m1fn),
#         int(TensorProto.INT32),
#         "TensorProto.FLOAT4E2M1",
#     ),
#     int(TensorProto.FLOAT8E8M0): TensorDtypeMap(
#         np.dtype(ml_dtypes.float8_e8m0fnu),
#         int(TensorProto.INT32),
#         "TensorProto.FLOAT8E8M0",
#     ),
#     int(TensorProto.UINT2): TensorDtypeMap(
#         np.dtype(ml_dtypes.uint2), int(TensorProto.INT32), "TensorProto.UINT2"
#     ),
#     int(TensorProto.INT2): TensorDtypeMap(
#         np.dtype(ml_dtypes.int2), int(TensorProto.INT32), "TensorProto.INT2"
#     ),
# }


def plc_type_from_dtype(dtype: str) -> str:
    """Map data type strings to appropriate PLC data types."""
    if dtype is None:
        raise ValueError("IR layer data type is None; tensor_info is incomplete.")

    match dtype:
        case "TensorProto.FLOAT":
            return "REAL"
        case "TensorProto.DOUBLE":
            return "LREAL" # TODO: this might be Double or something else, seems to be different across PLCs
        case "TensorProto.INT32":
            return "DINT"
        case "TensorProto.INT64":
            return "LINT"
        case _:
            logging.warning(f"Data type {dtype} is not explicitly supported, adding placeholder.")
            return "<UNSUPPORTED_TYPE>"

def generate_header(fb_name: str) -> STCode:
    """Generate function block header."""
    return STCode.from_lines(f"FUNCTION_BLOCK {fb_name}", "")


def generate_footer() -> STCode:
    """Generate function block footer."""
    return STCode.from_lines("END_FUNCTION_BLOCK", "")


def generate_var_input(network: NetworkIR) -> STCode:
    """Generate VAR_INPUT section."""
    if not network.layers:
        raise ValueError("Network has no layers defined.")

    first_layer = network.layers[0]
    if not hasattr(first_layer, "input_type"):
        raise ValueError("First layer does not have input_type attribute.")

    input_type = plc_type_from_dtype(first_layer.input_type)

    return STCode.from_lines(
        "VAR_INPUT",
        f"    input_data : ARRAY[0..{network.input_size-1}] OF {input_type};",
        "END_VAR",
        "",
    )


def generate_var_output(network: NetworkIR) -> STCode:
    """Generate VAR_OUTPUT section."""
    if not network.layers:
        raise ValueError("Network has no layers defined.")

    last_layer = network.layers[-1]
    if not hasattr(last_layer, "output_type"):
        raise ValueError("Last layer does not have output_type attribute.")

    output_type = plc_type_from_dtype(last_layer.output_type)

    return STCode.from_lines(
        "VAR_OUTPUT",
        f"    output_data : ARRAY[0..{network.output_size-1}] OF {output_type};",
        "END_VAR",
        "",
    )


def generate_weight_constant(layer) -> STCode:
    """Generate weight constant declaration for a layer as flattened 1D array."""
    weight_name = f"weights_{layer.layer_id}"
    weight_type = plc_type_from_dtype(layer.input_type)

    total_size = layer.input_size * layer.output_size
    flat_weights = layer.weights.flatten()
    weight_values = ", ".join(f"{val:.6f}" for val in flat_weights)

    return STCode.from_lines(
        f"{weight_name} : ARRAY[0..{total_size-1}] OF {weight_type} := [{weight_values}];",
    )


def generate_bias_constant(layer) -> STCode:
    """Generate bias constant declaration for a layer."""
    bias_name = f"bias_{layer.layer_id}"
    bias_type = plc_type_from_dtype(layer.output_type)
    bias_values = ", ".join(f"{layer.bias[j]:.6f}" for j in range(layer.bias.shape[0]))

    return STCode.from_lines(
        f"{bias_name} : ARRAY[0..{layer.output_size-1}] OF {bias_type} := [{bias_values}];"
    )


def generate_constants_section(network: NetworkIR) -> STCode:
    """Generate VAR CONSTANT section (weights, biases, ...)."""
    code = STCode.from_lines("VAR CONSTANT")

    for layer in network.layers:
        has_constants = False

        if hasattr(layer, "weights") and layer.weights is not None:
            code += generate_weight_constant(layer).indent()
            has_constants = True

        if hasattr(layer, "bias") and layer.bias is not None:
            code += generate_bias_constant(layer).indent()
            has_constants = True

        if has_constants:
            code += STCode.blank_line()

    code += STCode.from_lines("END_VAR", "")
    return code


def generate_layer_output_variables(layer) -> STCode:
    """Generate only output variable declarations for a single layer."""
    plc_type = plc_type_from_dtype(layer.output_type)

    return STCode.from_lines(
        f"(* Layer {layer.layer_id} output variable *)",
        f"layer_{layer.layer_id}_output : ARRAY[0..{layer.output_size-1}] OF {plc_type};",
    )


def generate_var_section(network: NetworkIR) -> STCode:
    """Generate VAR section with all internal variables."""
    code = STCode.from_lines("VAR")

    # Layer output variables
    for layer in network.layers:
        code += generate_layer_output_variables(layer).indent()
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
        getattr(layer, "activation", None) == ActivationType.SOFTMAX
        for layer in network.layers
    )
    if has_softmax:
        code += STCode.from_lines("    max_val : REAL;", "    exp_sum : REAL;")

    code += STCode.blank_line()
    code += STCode.from_lines("END_VAR", "")
    return code


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
    """Generate MatMul layer code: Y = A * B"""
    builder = STCodeBuilder()

    builder.add_line(f"(* Layer {layer.layer_id}: MatMul *)")
    builder.add_line(f"FOR j := 0 TO {layer.output_size-1} DO")
    with builder.indent():
        builder.add_line("sum := 0.0;")
        builder.add_line(f"FOR i := 0 TO {layer.input_size-1} DO")
        with builder.indent():
            # Use flattened array indexing: weights[i * output_size + j]
            builder.add_line(
                f"sum := sum + {input_var}[i] * weights_{layer.layer_id}[i * {layer.output_size} + j];"
            )
        builder.add_line("END_FOR;")
        builder.add_line(f"{output_var}[j] := sum;")
    builder.add_line("END_FOR;")

    return builder.build()


def generate_gemm_code(layer: GemmLayer, input_var: str, output_var: str) -> STCode:
    """Generate Gemm layer code: Y = alpha * A * B + beta * C"""
    builder = STCodeBuilder()

    builder.add_line(f"(* Layer {layer.layer_id}: Gemm *)")
    builder.add_line(f"FOR j := 0 TO {layer.output_size-1} DO")
    with builder.indent():
        builder.add_line("sum := 0.0;")
        builder.add_line(f"FOR i := 0 TO {layer.input_size-1} DO")
        with builder.indent():
            # Use flattened array indexing: weights[i * output_size + j]
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
    # TODO: Optimize by fusing the operations
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


# Mapping from layer type to code generator.
LAYER_CODE_GENERATORS = {
    MatMulLayer: generate_matmul_code,
    AddLayer: generate_add_code,
    GemmLayer: generate_gemm_code,
    FusedGemmLayer: generate_fused_gemm_code,
    ReshapeLayer: generate_reshape_code,
    ActivationLayer: generate_activation_layer_code
}


def generate_layer_computation(layer, input_var: str, output_var: str) -> STCode:
    """Generate computation code for a single layer."""
    for layer_type in type(layer).__mro__:
        if layer_type in LAYER_CODE_GENERATORS:
            return LAYER_CODE_GENERATORS[layer_type](layer, input_var, output_var)
        
    # no matching code generators:
    logger.error(
        f"Layer type {type(layer).__name__} has no matching code generator."
    )
    raise NotImplementedError(f"No code generator for layer type: {type(layer).__name__}")


def generate_forward_pass(network: NetworkIR) -> STCode:
    """Generate complete forward pass computation."""
    code = STCode.from_lines("(* Forward pass computation *)")
    current_input = "input_data"

    for idx, layer in enumerate(network.layers):
        is_last = idx == len(network.layers) - 1
        output_var = "output_data" if is_last else f"layer_{layer.layer_id}_output"

        code += generate_layer_computation(layer, current_input, output_var)
        code += STCode.blank_line()

        current_input = output_var

    return code


def generate_function_block(
    network: NetworkIR, fb_name: str = "NeuralNetwork"
) -> STCode:
    """Generate complete function block code."""
    logger.info(
        f"Generating function block '{fb_name}' with {len(network.layers)} layers"
    )

    code = STCode.empty()
    code += generate_header(fb_name)
    code += generate_var_input(network)
    code += generate_var_output(network)
    code += generate_constants_section(network)
    code += generate_var_section(network)
    code += generate_forward_pass(network)
    code += generate_footer()

    logger.info(f"Generated {len(code.lines)} lines of ST code.")
    return code
