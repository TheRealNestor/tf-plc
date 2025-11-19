"""
Main entry for the ONNX-to-ST code generator.

This file integrates the ONNX to IR transformation and the IR to ST generation.
"""

import argparse
from pathlib import Path
from .onnx_model import load_and_analyze_onnx_model
from .onnx_to_ir import onnx_to_ir
from .ir_to_st import generate_function_block
from .openplc_st import generate_program_wrapper, generate_openplc_configuration
from .onnx_model import ONNXModel

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def onnx_to_structured_text(
    analyzer: ONNXModel,
    fb_name: str = "NeuralNetworkFB",
    openplc: bool = False,
) -> str:
    """Convert ONNX model to Structured Text code."""
    network_ir = onnx_to_ir(analyzer)
    full_code = generate_function_block(network_ir, fb_name)

    if openplc:
        full_code = (
            full_code
            + generate_program_wrapper(
                fb_name, program_name="prog0", instance_name="nn"
            )
            + generate_openplc_configuration(
                program_name="prog0", instance_name="Inst0"
            )
        )

    return full_code.to_string()


def generate_st_from_onnx_file(
    onnx_path: Path,
    output_path: Path | None = None,
    fb_name: str = "NeuralNetworkFB",
    openplc: bool = False,
) -> str:
    """Generate Structured Text code from an ONNX file."""
    try:
        logger.info(f"Loading ONNX model from: {onnx_path}")
        analyzer = load_and_analyze_onnx_model(str(onnx_path))

        if not analyzer:
            raise ValueError("Failed to load ONNX model")

        logger.info("Generating Structured Text code...")
        code = onnx_to_structured_text(analyzer, fb_name=fb_name, openplc=openplc)

        if output_path is None:
            st_output_dir = Path("examples") / "models" / "structured_text"
            st_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = st_output_dir / f"{onnx_path.stem}.st"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(code)

        logger.info(f"Structured Text code saved to: {output_path}")
        logger.info(f"Total lines: {len(code.splitlines())}")

        return code

    except Exception as e:
        logger.error(f"Error generating Structured Text code: {e}", exc_info=True)
        raise


def main():
    """Command-line interface for the code generator."""
    parser = argparse.ArgumentParser(
        description="Convert ONNX neural network models to PLC Structured Text code"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to ONNX model file (.onnx)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output ST file path (default: models/structured_text/<name>.st)",
    )
    parser.add_argument(
        "--openplc",
        action="store_true",
        help="Generate OpenPLC wrapper code",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.input.exists():
        logger.error(f"File not found: {args.input}")
        return 1

    try:
        logger.info("Starting code generation...")
        code = generate_st_from_onnx_file(
            args.input,
            output_path=args.output,
            openplc=args.openplc,
        )

        logger.info(
            f"Successfully generated {len(code.splitlines())} lines of ST code"
        )
        return 0

    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
