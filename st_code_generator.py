"""
Functional Structured Text Code Generator for ONNX Models
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from onnx_model import ONNXModel


# ============================================================================
# Intermediate Representation (IR)
# ============================================================================

class ActivationType(Enum):
    """Supported activation functions"""
    NONE = "none"
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"


@dataclass(frozen=True)
class DenseLayer:
    """Represents a dense/fully-connected layer"""
    layer_id: int
    weights: np.ndarray
    bias: Optional[np.ndarray]
    activation: ActivationType
    input_size: int
    output_size: int


@dataclass(frozen=True)
class NetworkIR:
    """Intermediate representation of the neural network"""
    input_size: int
    output_size: int
    layers: Tuple[DenseLayer, ...]
    
    def __str__(self) -> str:
        return f"NetworkIR(input={self.input_size}, output={self.output_size}, layers={len(self.layers)})"


@dataclass(frozen=True)
class STCode:
    """Represents a piece of Structured Text code"""
    lines: Tuple[str, ...]

    def __add__(self, other: 'STCode') -> 'STCode':
        """Combine two code blocks"""
        return STCode(self.lines + other.lines)

    def indent(self, level: int = 1) -> 'STCode':
        """Return indented version of code"""
        indent_str = "    " * level
        return STCode(tuple(indent_str + line if line else line for line in self.lines))

    def to_string(self) -> str:
        """Convert to string"""
        return "\n".join(self.lines)

    @staticmethod
    def from_lines(*lines: str) -> 'STCode':
        """Create from individual lines"""
        return STCode(tuple(lines))

    @staticmethod
    def empty() -> 'STCode':
        """Create empty code block"""
        return STCode(())

    @staticmethod
    def blank_line() -> 'STCode':
        """Create blank line"""
        return STCode(("",))


# ============================================================================
# ONNX to IR Transformation
# ============================================================================

def parse_layer_activation(layers: List[Dict], start_idx: int) -> Tuple[ActivationType, int]:
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
    
    # Check for direct activation after MatMul
    match next_layer['op_type']:
        case 'Relu':
            return (ActivationType.RELU, 1)
        case 'Sigmoid':
            return (ActivationType.SIGMOID, 1)
        case 'Tanh':
            return (ActivationType.TANH, 1)
        case 'Softmax':
            return (ActivationType.SOFTMAX, 1)
        case 'Add':
            # Bias add - check next layer for activation
            if start_idx + 1 < len(layers):
                activation_layer = layers[start_idx + 1]
                match activation_layer['op_type']:
                    case 'Relu':
                        return (ActivationType.RELU, 2)  # Consumed Add + Relu
                    case 'Sigmoid':
                        return (ActivationType.SIGMOID, 2)  # Consumed Add + Sigmoid
                    case 'Tanh':
                        return (ActivationType.TANH, 2)  # Consumed Add + Tanh
                    case 'Softmax':
                        return (ActivationType.SOFTMAX, 2)  # Consumed Add + Softmax
                    case _:
                        return (ActivationType.NONE, 1)  # Just consumed Add, no activation
            return (ActivationType.NONE, 1)  # Just Add, no activation after
        case _:
            return (ActivationType.NONE, 0)


def extract_dense_layer(layer: Dict, layer_id: int, weights: Dict[str, np.ndarray], 
                        layers: List[Dict], layer_idx: int) -> Tuple[Optional[DenseLayer], int]:
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
    match layer['op_type']:
        case 'Gemm' | 'MatMul':
            # Find weight tensor from MatMul inputs
            weight_tensor = None
            
            for input_name in layer['inputs']:
                if input_name in weights:
                    tensor = weights[input_name]
                    if len(tensor.shape) == 2:
                        weight_tensor = tensor
                        break
            
            if weight_tensor is None:
                return (None, 1)
            
            input_size, output_size = weight_tensor.shape
            
            # Look ahead for bias (Add operation) and activation
            bias_tensor = None
            consumed = 1  # At least consumed the MatMul
            
            # Check if next layer is Add (bias)
            if layer_idx + 1 < len(layers) and layers[layer_idx + 1]['op_type'] == 'Add':
                add_layer = layers[layer_idx + 1]
                # Find bias tensor from Add inputs
                for input_name in add_layer['inputs']:
                    if input_name in weights:
                        tensor = weights[input_name]
                        if len(tensor.shape) == 1:
                            bias_tensor = tensor
                            break
            
            # Look ahead for activation (considering Add layer if present)
            activation, activation_layers = parse_layer_activation(layers, layer_idx + 1)
            consumed += activation_layers
            
            return (
                DenseLayer(
                    layer_id=layer_id,
                    weights=weight_tensor,
                    bias=bias_tensor,
                    activation=activation,
                    input_size=input_size,
                    output_size=output_size
                ),
                consumed
            )
        case _:
            print(f"[WARNING] Unsupported ONNX layer encountered at index {layer_idx}: '{layer['op_type']}'")
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
    # Extract input/output sizes
    input_info = list(analyzer.input_info.values())[0]
    output_info = list(analyzer.output_info.values())[0]
    
    input_shape = input_info['shape']
    output_shape = output_info['shape']
    
    # Handle batch dimension
    input_size = input_shape[1] if (input_shape[0] == -1 or input_shape[0] == 1) else input_shape[0]
    output_size = output_shape[1] if (output_shape[0] == -1 or output_shape[0] == 1) else output_shape[0]
    
    # Extract layers
    dense_layers = []
    layer_id = 0
    idx = 0
    
    while idx < len(analyzer.layers):
        layer, consumed = extract_dense_layer(
            analyzer.layers[idx],
            layer_id,
            analyzer.weights,
            analyzer.layers,
            idx
        )
        
        if layer:
            dense_layers.append(layer)
            layer_id += 1
        
        idx += consumed
    
    return NetworkIR(
        input_size=input_size,
        output_size=output_size,
        layers=tuple(dense_layers)
    )


# ============================================================================
# IR to Structured Text Code Generation
# ============================================================================

def generate_header(fb_name: str) -> STCode:
    """Generate function block header."""
    return STCode.from_lines(
        f"FUNCTION_BLOCK {fb_name}",
        ""
    )


def generate_footer() -> STCode:
    """Generate function block footer."""
    return STCode.from_lines("END_FUNCTION_BLOCK")


def generate_var_input(network: NetworkIR) -> STCode:
    """Generate VAR_INPUT section."""
    return STCode.from_lines(
        "VAR_INPUT",
        f"    input_data : ARRAY[0..{network.input_size-1}] OF REAL;",
        "END_VAR",
        ""
    )


def generate_var_output(network: NetworkIR) -> STCode:
    """Generate VAR_OUTPUT section."""
    return STCode.from_lines(
        "VAR_OUTPUT",
        f"    output_data : ARRAY[0..{network.output_size-1}] OF REAL;",
        "END_VAR",
        ""
    )


def generate_layer_variables(layer: DenseLayer) -> STCode:
    """Generate variable declarations for a single layer."""
    lines = [
        f"(* Layer {layer.layer_id} variables *)",
        f"weights_{layer.layer_id} : ARRAY[0..{layer.input_size-1}, 0..{layer.output_size-1}] OF REAL;",
    ]
    
    if layer.bias is not None:
        lines.append(f"bias_{layer.layer_id} : ARRAY[0..{layer.output_size-1}] OF REAL;")
    
    lines.append(f"layer_{layer.layer_id}_output : ARRAY[0..{layer.output_size-1}] OF REAL;")
    
    return STCode.from_lines(*lines)


def generate_var_section(network: NetworkIR) -> STCode:
    """Generate VAR section with all internal variables."""
    layer_vars = STCode.empty()
    for layer in network.layers:
        layer_vars = layer_vars + generate_layer_variables(layer) + STCode.blank_line()
    
    # Check if any layer uses softmax activation
    has_softmax = any(layer.activation == ActivationType.SOFTMAX for layer in network.layers)
    
    temp_vars_lines = [
        "(* Temporary computation variables *)",
        "i, j : INT;",
        "sum : REAL;",
    ]
    
    if has_softmax:
        temp_vars_lines.extend([
            "max_val : REAL;",
            "exp_sum : REAL;",
        ])
    
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
        "(* Initialize weights (one-time setup) *)",
        "IF NOT initialized THEN"
    )
    footer = STCode.from_lines(
        "    initialized := TRUE;",
        "END_IF;",
        ""
    )
    
    return header + init_code.indent() + footer


def generate_activation_code(activation: ActivationType, array_name: str, size: int) -> STCode:
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
                "END_FOR;"
            )
        
        case ActivationType.SIGMOID:
            return STCode.from_lines(
                "(* Sigmoid activation *)",
                f"FOR i := 0 TO {size-1} DO",
                f"    {array_name}[i] := 1.0 / (1.0 + EXP(-{array_name}[i]));",
                "END_FOR;"
            )
        
        case ActivationType.TANH:
            return STCode.from_lines(
                "(* Tanh activation *)",
                f"FOR i := 0 TO {size-1} DO",
                f"    {array_name}[i] := (EXP({array_name}[i]) - EXP(-{array_name}[i])) / (EXP({array_name}[i]) + EXP(-{array_name}[i]));",
                "END_FOR;"
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
                "END_FOR;"
            )
        
        case _:
            return STCode.empty()


def generate_matmul_code(layer: DenseLayer, input_var: str, output_var: str) -> STCode:
    """Generate matrix multiplication code for a dense layer."""
    has_bias = layer.bias is not None
    
    bias_line = f"{output_var}[j] := sum + bias_{layer.layer_id}[j];" if has_bias else f"{output_var}[j] := sum;"
    
    return STCode.from_lines(
        f"(* Layer {layer.layer_id}: Dense (MatMul + Bias) *)",
        f"FOR j := 0 TO {layer.output_size-1} DO",
        "    sum := 0.0;",
        f"    FOR i := 0 TO {layer.input_size-1} DO",
        f"        sum := sum + {input_var}[i] * weights_{layer.layer_id}[i,j];",
        "    END_FOR;",
        f"    {bias_line}",
        "END_FOR;"
    )


def generate_layer_forward_pass(layer: DenseLayer, input_var: str, 
                                is_last: bool) -> Tuple[STCode, str]:
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
    activation_code = generate_activation_code(layer.activation, output_var, layer.output_size)
    
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
        is_last = (idx == len(network.layers) - 1)
        layer_code, output_var = generate_layer_forward_pass(layer, current_input, is_last)
        forward_code = forward_code + layer_code
        current_input = output_var
    
    return header + forward_code


def generate_function_block(network: NetworkIR, fb_name: str = "NeuralNetwork") -> STCode:
    """
    Generate complete function block from network IR.
    
    This is the main composition function that combines all parts.
    """
    return (
        generate_header(fb_name) +
        generate_var_input(network) +
        generate_var_output(network) +
        generate_var_section(network) +
        generate_weight_initialization(network) +
        generate_forward_pass(network) +
        generate_footer()
    )


def generate_program_wrapper(
    fb_name: str, program_name: str = "prog0", instance_name: str = "nn"
) -> STCode:
    """Generate a PROGRAM wrapper that instantiates and calls the function block."""
    return STCode.from_lines(
        f"PROGRAM {program_name}",
        "VAR",
        f"    {instance_name} : {fb_name};",
        "END_VAR",
        "",
        f"{instance_name}();",
        "",
        "END_PROGRAM",
        "",
    )


def generate_openplc_configuration(
    program_name: str = "prog0",
    configuration_name: str = "Config0",
    resource_name: str = "Res0",
    task_name: str = "Main",
    task_interval: str = "T#1000ms",
    task_priority: int = 0,
    instance_name: str = "Inst0",
) -> STCode:
    """Generate OpenPLC configuration footer (CONFIGURATION / RESOURCE / TASK mapping)."""
    return STCode.from_lines(
        f"CONFIGURATION {configuration_name}",
        "",
        f"  RESOURCE {resource_name} ON PLC",
        f"    TASK {task_name}(INTERVAL := {task_interval},PRIORITY := {task_priority});",
        f"    PROGRAM {instance_name} WITH {task_name} : {program_name};",
        "  END_RESOURCE",
        "END_CONFIGURATION",
        "",
    )


# ============================================================================
# Main Pipeline
# ============================================================================


def onnx_to_structured_text(
    analyzer: ONNXModel,
    fb_name: str = "NeuralNetwork",
    include_program: bool = False,
    include_openplc_config: bool = False,
    program_name: str = "prog0",
    program_instance_name: str = "nn",
    cfg_instance_name: str = "Inst0",
) -> str:
    """
    Complete pipeline: ONNX -> IR -> Structured Text.

    By default this returns only the FUNCTION_BLOCK source. Set include_program
    and/or include_openplc_config to append a PROGRAM wrapper and OpenPLC configuration.
    """
    network_ir = onnx_to_ir(analyzer)
    fb_code = generate_function_block(network_ir, fb_name)

    full_code = fb_code
    if include_program:
        full_code = full_code + generate_program_wrapper(
            fb_name, program_name=program_name, instance_name=program_instance_name
        )

    if include_openplc_config:
        full_code = full_code + generate_openplc_configuration(
            program_name=program_name, instance_name=cfg_instance_name
        )

    return full_code.to_string()


def generate_st_from_onnx_file(
    onnx_path: str,
    output_path: str = None,
    fb_name: str = "NeuralNetwork",
    include_program: bool = False,
    include_openplc_config: bool = False,
    program_name: str = "prog0",
    program_instance_name: str = "nn",
    cfg_instance_name: str = "Inst0",
) -> str:
    """
    Convenience function to generate ST code from ONNX file.

    The function writes the generated ST file. By default it writes only the function block.
    Use include_program/include_openplc_config to append wrapper/configuration.
    """
    from pathlib import Path
    from onnx_model import load_and_analyze_onnx_model

    print("Loading ONNX model...")
    analyzer = load_and_analyze_onnx_model(onnx_path)

    if not analyzer:
        raise ValueError("Failed to load ONNX model")

    print("\nGenerating Structured Text code (functional approach)...")
    code = onnx_to_structured_text(
        analyzer,
        fb_name=fb_name,
        include_program=include_program,
        include_openplc_config=include_openplc_config,
        program_name=program_name,
        program_instance_name=program_instance_name,
        cfg_instance_name=cfg_instance_name,
    )

    # Auto-generate output path if not provided
    if output_path is None:
        onnx_path_obj = Path(onnx_path)
        st_output_dir = Path("models") / "structured_text"
        st_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = st_output_dir / f"{onnx_path_obj.stem}.st"

    with open(output_path, "w") as f:
        f.write(code)

    print(f"Structured Text code generated successfully: {output_path}")
    print(f"Total lines: {len(code.splitlines())}")

    return code


# ============================================================================
# Testing and Validation
# ============================================================================

def validate_network_ir(network: NetworkIR) -> List[str]:
    """
    Validate network IR for correctness.
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if network.input_size <= 0:
        errors.append(f"Invalid input size: {network.input_size}")
    
    if network.output_size <= 0:
        errors.append(f"Invalid output size: {network.output_size}")
    
    if len(network.layers) == 0:
        errors.append("Network has no layers")
    
    # Check layer connectivity
    expected_input = network.input_size
    for idx, layer in enumerate(network.layers):
        if layer.input_size != expected_input:
            errors.append(f"Layer {idx}: input size mismatch. Expected {expected_input}, got {layer.input_size}")
        expected_input = layer.output_size
    
    if expected_input != network.output_size:
        errors.append(f"Output size mismatch. Expected {network.output_size}, got {expected_input}")
    
    return errors


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert ONNX model to Structured Text")
    parser.add_argument("model_name", nargs="?", help="Name of the ONNX model file (without .onnx extension)")
    parser.add_argument("--with-program", action="store_true", help="Append PROGRAM wrapper to the generated file")
    parser.add_argument("--with-config", action="store_true", help="Append OpenPLC CONFIGURATION footer to the generated file")
    parser.add_argument("--program-name", default="prog0", help="PROGRAM name to use for wrapper/configuration")
    parser.add_argument("--program-instance", default="nn", help="Instance name for the PROGRAM wrapper (calls the FB)")
    parser.add_argument("--config-instance", default="Inst0", help="Instance name used in the CONFIGURATION PROGRAM line")
    args = parser.parse_args()

    # Find ONNX model
    models_dir = Path("models") / "onnx"

    if not models_dir.exists():
        print(f"ONNX models directory not found: {models_dir}")
        print("Please create the directory and convert a TensorFlow model to ONNX first")
        exit(1)

    onnx_models = list(models_dir.glob("*.onnx"))

    if not onnx_models:
        print(f"No ONNX models found in {models_dir}")
        print("Please convert a TensorFlow model to ONNX first")
        exit(1)

    # Select model based on CLI argument or use first one
    if args.model_name:
        model_path = models_dir / f"{args.model_name}.onnx"
        if not model_path.exists():
            print(f"Error: Model '{args.model_name}.onnx' not found in {models_dir}")
            print(f"\nAvailable models:")
            for model in onnx_models:
                print(f"  - {model.stem}")
            exit(1)
    else:
        model_path = onnx_models[0]
        print(f"No model specified, using: {model_path.stem}")

    # Create output directory for Structured Text files
    st_output_dir = Path("models") / "structured_text"
    st_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    model_name = model_path.stem
    output_path = st_output_dir / f"{model_name}.st"

    print(f"Converting {model_path} to Structured Text...")

    try:
        # Load and analyze
        from onnx_model import load_and_analyze_onnx_model
        analyzer = load_and_analyze_onnx_model(model_path)

        if analyzer:
            # Convert to IR
            print("\nConverting to intermediate representation...")
            network_ir = onnx_to_ir(analyzer)
            print(f"Network IR: {network_ir}")

            # Validate IR
            errors = validate_network_ir(network_ir)
            if errors:
                print("\nValidation errors:")
                for error in errors:
                    print(f"  - {error}")
            else:
                print("✓ Network IR validated successfully")

            # Generate code
            print("\nGenerating Structured Text code...")
            code = generate_st_from_onnx_file(
                model_path,
                output_path=output_path,
                fb_name="NeuralNetworkFB",
                include_program=args.with_program,
                include_openplc_config=args.with_config,
                program_name=args.program_name,
                program_instance_name=args.program_instance,
                cfg_instance_name=args.config_instance,
            )

            # Save to file
            with open(output_path, 'w') as f:
                f.write(code)

            print(f"\n{'='*60}")
            print("Generated code preview (first 50 lines):")
            print(f"{'='*60}")
            lines = code.split('\n')
            for line in lines[:50]:
                print(line)

            if len(lines) > 50:
                print(f"\n... ({len(lines) - 50} more lines)")

            print(f"\n✓ Code saved to: {output_path}")

    except Exception as e:
        print(f"Error generating Structured Text code: {e}")
        import traceback
        traceback.print_exc()
