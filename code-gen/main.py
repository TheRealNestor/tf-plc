"""
Main entry point for the ONNX to Structured Text code generator.

This script integrates the ONNX to IR transformation and the IR to ST generation,
and provides a command-line interface for user interaction.
"""

import argparse
from pathlib import Path
from .onnx_model import load_and_analyze_onnx_model
from .onnx_to_ir import onnx_to_ir
from .ir_to_st import generate_function_block
from .openplc_st import generate_program_wrapper, generate_openplc_configuration
from .onnx_model import ONNXModel


def onnx_to_structured_text(
    analyzer: ONNXModel,
    fb_name: str = "NeuralNetwork",
    openplc: bool = False,
    program_name: str = "prog0",
    program_instance_name: str = "nn",
    cfg_instance_name: str = "Inst0",
) -> str:
    network_ir = onnx_to_ir(analyzer)
    full_code = generate_function_block(network_ir, fb_name)

    if openplc:
        full_code = (
            full_code
            + generate_program_wrapper(
                fb_name, program_name=program_name, instance_name=program_instance_name
            )
            + generate_openplc_configuration(
                program_name=program_name, instance_name=cfg_instance_name
            )
        )

    return full_code.to_string()


def generate_st_from_onnx_file(
    onnx_path: str,
    output_path: str = None,
    fb_name: str = "NeuralNetwork",
    openplc: bool = False,
    program_name: str = "prog0",
    program_instance_name: str = "nn",
    cfg_instance_name: str = "Inst0",
) -> str:
    from pathlib import Path
    from .onnx_model import load_and_analyze_onnx_model

    print("Loading ONNX model...")
    analyzer = load_and_analyze_onnx_model(onnx_path)

    if not analyzer:
        raise ValueError("Failed to load ONNX model")

    print("\nGenerating Structured Text code...")
    code = onnx_to_structured_text(
        analyzer,
        fb_name=fb_name,
        openplc=openplc,
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to Structured Text"
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        help="Name of the ONNX model file (without .onnx extension)",
    )
    parser.add_argument(
        "--openplc",
        action="store_true",
        help="Generate OpenPLC PROGRAM wrapper and CONFIGURATION",
    )
    args = parser.parse_args()

    models_dir = Path("models") / "onnx"
    st_output_dir = Path("models") / "structured_text"
    st_output_dir.mkdir(parents=True, exist_ok=True)

    onnx_models = list(models_dir.glob("*.onnx"))
    if not onnx_models:
        print(f"No ONNX models found in {models_dir}")
        exit(1)

    if args.model_name:
        model_path = models_dir / f"{args.model_name}.onnx"
        if not model_path.exists():
            print(f"Error: Model '{args.model_name}.onnx' not found in {models_dir}")
            exit(1)
    else:
        model_path = onnx_models[0]
        print(f"No model specified, using: {model_path.stem}")

    output_path = st_output_dir / f"{model_path.stem}.st"

    try:
        code = generate_st_from_onnx_file(
            model_path,
            output_path=output_path,
            fb_name="NeuralNetworkFB",
            openplc=args.openplc,
            program_name="prog0",  # Hardcoded
            program_instance_name="nn",  # Hardcoded
            cfg_instance_name="Inst0",  # Hardcoded
        )
        print(f"\n✓ Code saved to: {output_path}")
    except Exception as e:
        print(f"Error generating Structured Text code: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
