#!/usr/bin/bash

filename="test7"

# We use tee to write to the file and the screen simultaneously
python "examples/models/local_model.py" "$filename" | \
tee /dev/tty | \
sed -n '/Model: "sequential"/,$p' > "examples/models/structured_text/$filename.summary"

python src/util/convert_to_onnx.py keras "examples/models/keras/$filename.keras" "examples/models/onnx/$filename.onnx"

python src/codegen/main.py "examples/models/onnx/$filename.onnx"
