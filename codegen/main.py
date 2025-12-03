"""
Main code generation entry point.
"""

import logging
from pathlib import Path
import argparse

from .onnx_model import ONNXModel
from .onnx_to_ir import onnx_to_ir
from .ir_to_st import translate_ir_to_st

logger = logging.getLogger(__name__)


def onnx_to_structured_text(
    model_path: str,
    output_path: str | None = None,
    block_name: str = "NeuralNetworkFB",
) -> str:
    """
    Convert ONNX model to IEC 61131-3 Structured Text.
    // ...existing code...
    """
    logger.info("Starting code generation...")

    # Load and analyze ONNX model
    logger.info(f"Loading ONNX model from: {model_path}")
    analyzer = ONNXModel(model_path)

    if not analyzer.load_model():
        raise RuntimeError(f"Failed to load ONNX model from {model_path}")

    logger.info("Generating Structured Text code...")
    ir = onnx_to_ir(analyzer)
    st_code = translate_ir_to_st(ir, block_name)

    # Save if output path provided
    if output_path:
        Path(output_path).write_text(st_code, encoding="utf-8")
        logger.info(f"Structured Text code saved to: {output_path}")

    line_count = len(st_code.splitlines())
    logger.info(f"Successfully generated {line_count} lines of ST code")

    return st_code


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert ONNX models to IEC 61131-3 Structured Text"
    )
    parser.add_argument("model_path", type=str, help="Path to ONNX model file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for generated ST code (default: auto-generate from model path)",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="NeuralNetworkFB",
        help="Name for the generated function block (default: NeuralNetworkFB)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s:%(name)s:%(message)s")

    # Auto-generate output path if not provided
    output_path = args.output
    if output_path is None:
        model_path = Path(args.model_path)
        output_path = str(
            model_path.parent.parent / "structured_text" / f"{model_path.stem}.st"
        )
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Run conversion
    try:
        onnx_to_structured_text(
            model_path=args.model_path, output_path=output_path, block_name=args.name
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=args.verbose)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
