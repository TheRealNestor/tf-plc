# tf-plc

Tool suite for converting ONNX neural network models to IEC 61131-3 Structured Text (ST) for PLCs.

## Features

- Inspect and summarize ONNX models (weights, layers, inputs/outputs)
- Convert ONNX models to intermediate representation (IR)
- Generate Structured Text code for feedforward neural networks
- Validate translation by converting ST code back to Python and comparing outputs

## Usage

1. Place ONNX models in `examples/models/onnx/`
2. Run the main compiler:

   ```
   python src/codegen/main.py examples/models/onnx/your_model.onnx
   ```

   This generates ST code from the ONNX model.

3. Optional: Validate translation (was used to ensure correctness of the code generator)
   ```
   python src/translation_validation/validation.py path/to/generated.st path/to/save.py
   ```

## Installation

**Windows:**

```sh
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

**Linux/macOS:**

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Troubleshooting

- Ensure ONNX files exist and are valid
- Only simple feedforward networks such as MLPs have been tested and are fully supported. The underlying IR design is easily extended to other types of networks.
- Check logs for unsupported operators or errors

