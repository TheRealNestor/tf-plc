"""
Main entry point for ONNX to Structured Text compiler.
"""

import logging
import argparse
import sys
from pathlib import Path

from codegen.onnx_model import ONNXModel
from codegen.onnx_to_ir import onnx_to_ir
from codegen.ir_optimizer import IROptimizer
from codegen.ir_to_st import translate_ir_to_st

logger = logging.getLogger(__name__)


def compile_onnx_to_st(
    model_path: str, optimize: bool = True, output_path: str = None
) -> str:
    """
    Complete compilation pipeline: ONNX → IR → Optimized IR → ST Code

    Args:
        model_path: Path to ONNX model file
        optimize: Whether to apply optimization passes
        output_path: Optional path to save generated ST code

    Returns:
        Generated Structured Text code as string
    """
    logger.info(f"Compiling ONNX model: {model_path}")

    # Step 1: Load and analyze ONNX model
    logger.info("Step 1: Loading ONNX model...")
    analyzer = ONNXModel(model_path)
    analyzer.load_model()

    # Step 2: Convert to IR (complete, unoptimized)
    logger.info("Step 2: Converting to IR...")
    ir = onnx_to_ir(analyzer)
    logger.info(f"  Created IR with {len(ir.layers)} layers")

    # Step 3: Optimize IR (optional)
    if optimize:
        logger.info("Step 3: Optimizing IR...")
        optimizer = IROptimizer(ir)
        ir = optimizer.optimize()  # Uses DEFAULT_PASSES
        logger.info(f"  Optimized to {len(ir.layers)} layers")
    else:
        logger.info("Step 3: Skipping optimization (optimize=False)")

    # Step 4: Generate Structured Text code
    logger.info("Step 4: Generating Structured Text code...")
    st_code = translate_ir_to_st(ir, fb_name="NeuralNetworkFB")

    # Step 5: Save to file (optional)
    if output_path:
        logger.info(f"Step 5: Writing to {output_path}")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(st_code)

    logger.info("Compilation complete!")
    logger.info(f"Generated ST code lines: {len(st_code.splitlines())}")
    return st_code


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compile ONNX neural network models to IEC 61131-3 Structured Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic compilation
  python -m codegen.main model.onnx -o output.st

  # Without optimization
  python -m codegen.main model.onnx -o output.st --no-optimize

  # Verbose output
  python -m codegen.main model.onnx -o output.st -v

  # Auto-generate output filename
  python -m codegen.main model.onnx
        """,
    )

    parser.add_argument("input", type=str, help="Path to input ONNX model file")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output Structured Text file (default: <input_name>.st)",
    )

    parser.add_argument(
        "--no-optimize", action="store_true", help="Disable IR optimization passes"
    )

    parser.add_argument(
        "--fb-name",
        type=str,
        default="NeuralNetworkFB",
        help="Name for the generated function block (default: NeuralNetworkFB)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose/debug output"
    )

    parser.add_argument(
        "--version", action="version", version="ONNX to ST Compiler v0.1.0"
    )

    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()

    setup_logging(args.verbose)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if not input_path.suffix.lower() == ".onnx":
        logger.warning(f"Input file does not have .onnx extension: {args.input}")

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_suffix(".st")
        if input_path.parent.name.lower() == "onnx":
            output_path = input_path.parent.parent / "structured_text" / output_path.name

        logger.info(f"Auto-generated output path: {output_path}")

    try:
        compile_onnx_to_st(
            model_path=str(input_path),
            optimize=not args.no_optimize,
            output_path=str(output_path),
        )

        logger.info(f"Successfully compiled {input_path.name}")
        logger.info(f"Output written to {output_path}")

    except Exception as e:
        logger.error(f"Compilation failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
