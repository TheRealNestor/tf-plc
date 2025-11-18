# tf-plc

Tools to inspect ONNX neural networks and translate simple feed‑forward networks into PLC Structured Text (IEC 61131-3).

## Overview

This project contains utilities to:
- Inspect and summarize ONNX models (weights, layers, inputs/outputs).
- Generate Structured Text (ST) code that implements a simple feedforward neural network (ReLU activations) from an ONNX model.

## Key files

- `onnx_model.py`  
  - Provides `ONNXModel` class to load and analyze ONNX files.
  - Main capabilities:
    - load_model(): validate and load an ONNX model.
    - extract_weights(): extract initializers (weights/biases) as numpy arrays.
    - analyze_layers(): enumerate nodes, inputs/outputs and extract attributes.
    - get_input_output_info(): report model input/output shapes and dtypes.
    - print_model_summary(): prints a full model summary to stdout.
  - CLI usage (examples):
    - Analyze default model (first `.onnx` in `models/onnx`):
      python onnx_model.py
    - Analyze a specific model by name (without .onnx):
      python onnx_model.py my_model
  - Output: human-readable model summary printed to the console; `ONNXModel` instance exposes `weights`, `layers`, `input_info`, `output_info` for programmatic use.

- `st_code_generator.py` (translator / generator)  
  - Translates supported ONNX graphs (simple feedforward with ReLU) into PLC Structured Text.
  - Typical usage:
      python st_code_generator.py path/to/model.onnx --out-dir generated_st
  - Output: ST source files (Function Blocks / Programs) that represent the network topology and parameter arrays.
  - Limitations: Designed for simple feedforward networks; many ONNX ops and dynamic graph constructs may not be supported. Review generated ST before deploying.

## Installation

1. Create and activate a Python virtual environment (Windows example):
   python -m venv .venv
   .venv\Scripts\activate

2. Install project
    - As editable (developer mode): `pip install -e .`
    - Normal install: `pip install .`

## Usage tips

- Place ONNX models in `models/onnx/` (the `onnx_model.py` CLI looks there by default).
- Use `onnx_model.py` to inspect the model before generating code to verify supported operators and shapes.
- Run `st_code_generator.py` after inspection to produce Structured Text.

## Troubleshooting

- "Model file not found" — ensure the file path is correct and the `.onnx` file exists.
- ONNX checker errors — the model may be invalid or contain unsupported ops; try exporting a simpler model.
- Generation issues — examine printed layer info and attributes to find unsupported operators.

## Contributing

- Open an issue with a minimal reproducible ONNX model if the generator fails on a simple feedforward net.
- Pull requests adding op support or improving codegen are welcome.

## License & Contact

- Check project root for a LICENSE file. For questions, open an issue in the repository.