import argparse
from pathlib import Path
import onnxruntime as ort


def optimize_onnx_model(input_model_path: Path, output_model_path: Path) -> None:
    """
    Optimize an ONNX model and save the optimized version.

    Args:
        input_model_path: Path to the input ONNX model file.
        output_model_path: Path to save the optimized ONNX model file.
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = str(output_model_path)

    session = ort.InferenceSession(str(input_model_path), sess_options)
    print(f"Optimized model saved to: {output_model_path}")

def main(): 
    parser = argparse.ArgumentParser(description="Optimize an ONNX model.")
    parser.add_argument("--input", "-i", required=False, help="Path to the input ONNX model file.")
    parser.add_argument("--output", "-o", required=False, help="Path to the output ONNX model file.")
    parser.add_argument("--model-dir", "-d", default="", help="Directory containing ONNX models")

    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    input_path = model_dir / args.input
    output_path = model_dir / args.output

    optimize_onnx_model(input_path, output_path)

if __name__ == "__main__":
    main()


