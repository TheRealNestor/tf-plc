from pathlib import Path
import argparse
import subprocess
import sys


def convert_saved_model_to_onnx(saved_model_path, output_path):
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--saved-model",
                str(saved_model_path),
                "--output",
                str(output_path),
            ],
            check=True,
        )
        print(f"Successfully converted saved model to ONNX: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting saved model to ONNX: {e}")


def convert_tflite_to_onnx(tflite_model_path, output_path):
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--tflite",
                str(tflite_model_path),
                "--output",
                str(output_path),
            ],
            check=True,
        )
        print(f"Successfully converted TFLite model to ONNX: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting TFLite model to ONNX: {e}")


def convert_keras_to_onnx(keras_model_path, output_path):
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--keras",
                str(keras_model_path),
                "--output",
                str(output_path),
            ],
            check=True,
        )
        print(f"Successfully converted Keras model to ONNX: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting Keras model to ONNX: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert models to ONNX format.")
    parser.add_argument(
        "model_type",
        choices=["saved_model", "tflite", "keras"],
        help="Type of the model to convert (saved_model, tflite, keras).",
    )
    parser.add_argument("input_path", help="Full path to the input model file.")
    parser.add_argument("output_path", help="Full path to the output ONNX file.")

    args = parser.parse_args()

    # Convert based on model type
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.model_type == "saved_model":
        convert_saved_model_to_onnx(input_path, output_path)
    elif args.model_type == "tflite":
        convert_tflite_to_onnx(input_path, output_path)
    elif args.model_type == "keras":
        convert_keras_to_onnx(input_path, output_path)
