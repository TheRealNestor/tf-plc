"""
Loads and analyzes ONNX models to extract weights, layer information, and model structure for code generation.
"""

import onnx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ONNXModel:
    """
    A class to load and analyze ONNX models, extracting weights, layer information,
    and model structure for later code generation.
    """

    def __init__(self, model_path: str | Path):
        """
        Initialize the analyzer with an ONNX model.

        Args:
            model_path: Path to the ONNX model file (string or Path object)
        """
        self.model_path = Path(model_path)
        self.model = None
        self.graph = None
        self.weights = {}
        self.layers = []
        self.input_info = {}
        self.output_info = {}
        self.tensor_info = {}  # Maps tensor names to their types and shapes

    def load_model(self) -> bool:
        """
        Load the ONNX model from file.

        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"ONNX model file not found: {self.model_path}")
                return False

            self.model = onnx.load(str(self.model_path))

            onnx.checker.check_model(self.model)
            logger.info(f"Successfully loaded ONNX model: {self.model_path}")

            self.graph = self.model.graph
            return True

        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            return False

    @staticmethod
    def get_tensor_type_name(elem_type: int) -> str:
        """Convert ONNX tensor type to readable string."""
        # Refer to: https://onnx.ai/onnx/intro/concepts.html#element-type
        return onnx.helper.tensor_dtype_to_string(elem_type)

    @staticmethod
    def parse_value(value: onnx.ValueInfoProto) -> Dict[str, Any]:
        """Extract dtype (as ONNX string) and shape from a ValueInfoProto."""
        t = value.type.tensor_type
        elem = t.elem_type

        onnx_type = onnx.helper.tensor_dtype_to_string(elem)

        shape = []
        for d in t.shape.dim:
            if d.dim_value > 0:  # Fixed dimension
                shape.append(d.dim_value)
            elif d.dim_param:  # Symbolic dimension
                shape.append(str(d.dim_param))
            else:  # Unknown dimension
                shape.append(None)

        return {
            "onnx_type": onnx_type,
            "shape": shape,
        }

    def _build_tensor_info(self):
        """Build tensor_info, input_info, and output_info using ONNX shape inference."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        try:
            inferred = onnx.shape_inference.infer_shapes(self.model)
        except Exception as e:
            logger.warning(f"Shape inference failed: {e}. Using raw graph.")
            inferred = self.model

        tensor_info = {}

        # Inputs (excluding initializers)
        initializer_names = {init.name for init in inferred.graph.initializer}

        for v in inferred.graph.input:
            if v.name not in initializer_names:
                tensor_info[v.name] = self.parse_value(v)

        # Outputs
        for v in inferred.graph.output:
            tensor_info[v.name] = self.parse_value(v)

        # Intermediate tensors
        for v in inferred.graph.value_info:
            tensor_info[v.name] = self.parse_value(v)

        self._infer_missing_tensor_info(tensor_info, inferred)

        # Fill member variables
        self.tensor_info = tensor_info
        self.input_info = {
            name: info
            for name, info in tensor_info.items()
            if any(inp.name == name for inp in self.graph.input)
        }
        self.output_info = {
            name: info
            for name, info in tensor_info.items()
            if any(out.name == name for out in self.graph.output)
        }

        # Ensure layers are also analyzed once
        if not self.layers:
            self.analyze_layers()
            self.extract_weights()

        logger.info(f"Extracted tensor info for {len(self.tensor_info)} tensors.")

    def _infer_missing_tensor_info(self, tensor_info: Dict, inferred_model):
        """
        Infer type and shape for intermediate tensors that ONNX shape inference missed.
        This happens when layers output tensors that aren't in value_info.

        Args:
            tensor_info: Dictionary to populate with inferred info (modified in-place)
            inferred_model: ONNX model after shape inference
        """
        # Make sure weights are extracted first
        if not self.weights:
            self.extract_weights()

        # Process nodes in order to propagate shape information
        for node in inferred_model.graph.node:
            # Check each output of this node
            for output_name in node.output:
                if output_name in tensor_info:
                    continue  # Already have info

                # Try to infer from input tensor and operation
                if not node.input:
                    continue

                input_name = node.input[0]
                if input_name not in tensor_info:
                    logger.debug(
                        f"Cannot infer '{output_name}': input '{input_name}' missing from tensor_info"
                    )
                    continue

                input_info = tensor_info[input_name]

                # Infer based on operation type
                inferred_info = self._infer_tensor_from_op(
                    node, input_info, tensor_info
                )

                if inferred_info:
                    tensor_info[output_name] = inferred_info
                    logger.debug(
                        f"Inferred tensor '{output_name}' from {node.op_type}: "
                        f"type={inferred_info.get('onnx_type')}, "
                        f"shape={inferred_info.get('shape')}"
                    )
                else:
                    logger.debug(
                        f"Could not infer tensor '{output_name}' from {node.op_type}"
                    )

    def _infer_tensor_from_op(
        self, node, input_info: Dict, tensor_info: Dict
    ) -> Optional[Dict]:
        op_type = node.op_type
        input_type = input_info.get("onnx_type")
        if not input_type:
            logger.warning(f"Input tensor type unknown for operation {op_type}")
            raise ValueError("Input tensor type unknown")

        # Operations that preserve shape and type
        if op_type in ["Relu", "Sigmoid", "Tanh", "Softmax", "Add"]:
            return {
                "onnx_type": input_type,
                "shape": (
                    input_info.get("shape", []).copy()
                    if input_info.get("shape")
                    else []
                ),
            }

        # need to check weight shape for these
        if op_type in ["MatMul", "Gemm", "FusedGemm"]:
            # Find weight tensor (second input, should be in initializers)
            if len(node.input) < 2:
                return None

            weight_name = node.input[1]
            if weight_name not in self.weights:
                return None

            weight_shape = self.weights[weight_name].shape
            if len(weight_shape) != 2:
                return None

            # Output shape: [batch_size, output_features]
            input_shape = input_info.get("shape", [])
            batch_size = input_shape[0] if input_shape else 1
            output_features = weight_shape[
                1
            ]  # Assuming [input_features, output_features]

            return {"onnx_type": input_type, "shape": [batch_size, output_features]}

        # Reshape - check shape input
        if op_type == "Reshape":
            if len(node.input) < 2:
                return None

            shape_name = node.input[1]
            if shape_name not in self.weights:
                return None

            target_shape = self.weights[shape_name].tolist()
            return {"onnx_type": input_type, "shape": [int(d) for d in target_shape]}

        # QuantizeLinear/DequantizeLinear preserve shape
        # QuantizeLinear - outputs INT8/UINT8 typically
        if op_type == "QuantizeLinear":
            # Check if there's a zero_point to infer the output type
            output_type = "TensorProto.INT8"  # Common default for quantization
            if len(node.input) >= 3:
                zero_point_name = node.input[2]
                if zero_point_name in self.weights:
                    zero_point = self.weights[zero_point_name]
                    # Infer type from zero point dtype
                    if zero_point.dtype == np.uint8:
                        output_type = "TensorProto.UINT8"
                    elif zero_point.dtype == np.int8:
                        output_type = "TensorProto.INT8"

            return {
                "onnx_type": output_type,
                "shape": (
                    input_info.get("shape", []).copy()
                    if input_info.get("shape")
                    else []
                ),
            }

        if op_type == "DequantizeLinear":
            output_type = "TensorProto.FLOAT"  # Default

            # Try to infer from scale tensor (second input)
            if len(node.input) >= 2:
                scale_name = node.input[1]
                if scale_name in self.weights:
                    scale = self.weights[scale_name]
                    # Infer output type from scale dtype
                    if scale.dtype == np.float32:
                        output_type = "TensorProto.FLOAT"
                    elif scale.dtype == np.float64:
                        output_type = "TensorProto.DOUBLE"
                    elif scale.dtype == np.float16:
                        output_type = "TensorProto.FLOAT16"
                # If scale is in tensor_info instead of weights (dynamic scale)
                elif scale_name in tensor_info:
                    scale_type = tensor_info[scale_name].get("onnx_type")
                    if scale_type:
                        output_type = scale_type

            return {
                "onnx_type": output_type,
                "shape": (
                    input_info.get("shape", []).copy()
                    if input_info.get("shape")
                    else []
                ),
            }

        # Unknown operation type
        return None

    def extract_weights(self) -> Dict[str, np.ndarray]:
        """
        Extract all weights and biases from the model.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping parameter names to numpy arrays
        """
        if not self.model:
            logger.error("Model not loaded. Call load_model() first.")
            return {}

        weights = {}

        # Extract weights and biases
        for initializer in self.graph.initializer:
            tensor_data = onnx.numpy_helper.to_array(initializer)
            weights[initializer.name] = tensor_data

        self.weights = weights
        return weights

    def analyze_layers(self) -> List[Dict[str, Any]]:
        """
        Analyze all layers/nodes in the model.

        Returns:
            List[Dict[str, Any]]: List of layer information dictionaries
        """
        if not self.model:
            logger.error("Model not loaded. Call load_model() first.")
            return []

        if self.layers:
            return self.layers

        layers = []

        for node in self.graph.node:
            layer_info = {
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": {},
            }

            # Extract attributes
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    layer_info["attributes"][attr.name] = attr.i
                elif attr.type == onnx.AttributeProto.FLOAT:
                    layer_info["attributes"][attr.name] = attr.f
                elif attr.type == onnx.AttributeProto.STRING:
                    layer_info["attributes"][attr.name] = attr.s.decode("utf-8")
                elif attr.type == onnx.AttributeProto.INTS:
                    layer_info["attributes"][attr.name] = list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    layer_info["attributes"][attr.name] = list(attr.floats)

            layers.append(layer_info)

        self.layers = layers
        return layers

    def get_input_output_info(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get information about model inputs and outputs.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Input info and output info
        """
        if not self.model:
            logger.error("Model not loaded. Call load_model() first.")
            return {}, {}

        if self.input_info and self.output_info:
            return self.input_info, self.output_info

        input_info = {}
        for input_tensor in self.graph.input:
            if input_tensor.name not in [init.name for init in self.graph.initializer]:
                shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    else:
                        shape.append(-1)

                input_info[input_tensor.name] = {
                    "shape": shape,
                    "dtype": input_tensor.type.tensor_type.elem_type,
                }

        output_info = {}
        for output_tensor in self.graph.output:
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)

            output_info[output_tensor.name] = {
                "shape": shape,
                "dtype": output_tensor.type.tensor_type.elem_type,
            }

        self.input_info = input_info
        self.output_info = output_info
        return input_info, output_info

    def resolve_static_dims(self, shape, tensor_name):
        """
        Extract static positive integer dimensions. We cannot feasibly handle dynamic/symbolic dims in PLC.
        Raise failure if NO static dims exist or shape is purely symbolic.
        """
        static = [d for d in shape if isinstance(d, int) and d > 0]

        if not static:
            raise ValueError(
                f"Cannot determine static size of tensor '{tensor_name}'. "
                f"Shape={shape}. "
                f"This model uses symbolic or dynamic dimensions "
                f"which PLC Structured Text cannot represent."
            )

        return static

    def static_product(self, shape, tensor_name):
        """
        Compute product of static dims only.
        Raises an error if symbolic or unknown dims prevent determining a static size.
        """
        static = self.resolve_static_dims(shape, tensor_name)
        return int(np.prod(static))

    def get_tensor_size(self, tensor_name: str) -> int:
        """
        Get the size of a tensor from either weights (constants) or tensor_info (dynamic).

        Args:
            tensor_name: Name of the tensor
            analyzer: ONNX model analyzer

        Returns:
            int: Total number of elements in the tensor
        """
        # Check if it's a constant weight first
        if tensor_name in self.weights:
            size = self.weights[tensor_name].size
            logger.debug(f"Tensor '{tensor_name}' is a constant with size {size}")
            return size

        # Otherwise look in tensor_info for dynamic tensors
        if tensor_name in self.tensor_info:
            shape = self.tensor_info[tensor_name]["shape"]
            size = self.static_product(shape, tensor_name)
            logger.debug(f"Tensor '{tensor_name}' has shape {shape}, size {size}")
            return size

        # Not found anywhere
        raise ValueError(
            f"Cannot find tensor '{tensor_name}' in either weights or tensor_info. "
            f"Cannot determine size."
        )

    def get_quantized_weight(
        self, tensor_name: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get a weight tensor with its scale and zero-point.

        Args:
            tensor_name: Name of the dequantized tensor (output of DequantizeLinear)

        Returns:
            Tuple of (quantized_weight, scale, zero_point) if found, else None.
        """

        # Check if it is an output of a DequantizeLinear layer
        for layer in self.layers:
            if (
                layer["op_type"] == "DequantizeLinear"
                and layer["outputs"][0] == tensor_name
            ):
                quantized_name = layer["inputs"][0]
                scale_name = layer["inputs"][1]
                zero_point_name = layer["inputs"][2]

                if (
                    quantized_name in self.weights
                    and scale_name in self.weights
                    and zero_point_name in self.weights
                ):

                    return (
                        self.weights[quantized_name],
                        self.weights[scale_name],
                        self.weights[zero_point_name],
                    )

        return None

    def get_dequantized_weight(self, tensor_name: str) -> Optional[np.ndarray]:
        """
        Get a weight tensor, dequantizing if necessary.

        For quantized models, it attempts to trace back through DequantizeLinear layers
        to find the original quantized weights, scale, and zero-point.

        Args:
            tensor_name: Name of the weight tensor (could be dequantized)

        Returns:
            np.ndarray: Dequantized weight tensor, or None if not found
        """

        # It might already be a weight tensor / dequantized
        if tensor_name in self.weights:
            return self.weights[tensor_name]

        quantized_info = self.get_quantized_weight(tensor_name)

        if quantized_info is None:
            logger.warning(
                f"Weight tensor '{tensor_name}' not found as dequantized or quantized."
            )
            return None

        quantized_weight, scale, zero_point = quantized_info
        # dequantize: real_value = scale * (quantized_value - zero_point)
        dequantized = scale * (
            quantized_weight.astype(np.float32) - zero_point.astype(np.float32)
        )
        return dequantized

    def get_weights_and_biases(
        self, layer_inputs: List[str]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract weight matrix (2D) and bias vector (1D) from layer inputs.

        Args:
            layer_inputs: List of input tensor names

        Returns:
            Tuple of (weight_tensor, bias_tensor), either may be None
        """
        weight_tensor = None
        bias_tensor = None

        for name in layer_inputs:

            # Check for quantized weights first, since these may not be in weights directly
            tensor = self.get_dequantized_weight(name)

            if tensor is not None:
                if len(tensor.shape) == 2:
                    weight_tensor = tensor
                elif len(tensor.shape) == 1:
                    bias_tensor = tensor

        return weight_tensor, bias_tensor

    def print_model_summary(self):
        """Print a comprehensive summary of the model."""
        if not self.model:
            logger.error("Model not loaded. Call load_model() first.")
            return

        print("\n" + "=" * 60)
        print("ONNX MODEL SUMMARY")
        print("=" * 60)

        print(f"Model path: {self.model_path}")
        print(f"IR Version: {self.model.ir_version}")
        print(f"Producer: {self.model.producer_name} {self.model.producer_version}")

        input_info, output_info = self.get_input_output_info()
        print(f"\nInputs ({len(input_info)}):")
        for name, info in input_info.items():
            shape = info.get("shape")
            # Prefer onnx_type (string) if available, else fallback to dtype (int enum)
            dtype = info.get("onnx_type", info.get("dtype"))
            print(f"  - {name}: shape={shape}, dtype={dtype}")

        print(f"\nOutputs ({len(output_info)}):")
        for name, info in output_info.items():
            shape = info.get("shape")
            dtype = info.get("onnx_type", info.get("dtype"))
            print(f"  - {name}: shape={shape}, dtype={dtype}")

        layers = self.analyze_layers()
        print(f"\nLayers ({len(layers)}):")
        layer_types: dict[str, int] = {}
        for layer in layers:
            op = layer["op_type"]
            layer_types[op] = layer_types.get(op, 0) + 1
            print(
                f"  - {layer['name'] or '<unnamed>'}: "
                f"type={op}, inputs={layer['inputs']}, outputs={layer['outputs']}"
            )

        print("\nLayer type counts:")
        for op, count in sorted(layer_types.items()):
            print(f"  {op}: {count}")
        print("=" * 60)


def load_and_analyze_onnx_model(model_path: str | Path) -> ONNXModel:
    """
    Convenience function to load and analyze an ONNX model.

    Args:
        model_path: Path to the ONNX model file (string or Path object)

    Returns:
        ONNXModel: Loaded ONNX model
    """
    analyzer = ONNXModel(model_path)

    if analyzer.load_model():
        analyzer._build_tensor_info()
        analyzer.extract_weights()
        analyzer.analyze_layers()
        analyzer.print_model_summary()
        return analyzer
    else:
        return None


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load and analyze ONNX model")
    parser.add_argument(
        "model_name",
        nargs="?",
        help="Name of the ONNX model file (without .onnx extension)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    models_dir = Path("examples") / "models" / "onnx"

    if not models_dir.exists():
        logger.error(f"ONNX models directory not found: {models_dir}")
        exit(1)

    onnx_models = list(models_dir.glob("*.onnx"))

    if not onnx_models:
        logger.error(f"No ONNX models found in {models_dir}")
        exit(1)

    # Select model based on CLI argument or use first one
    if args.model_name:
        model_path = models_dir / f"{args.model_name}.onnx"
        if not model_path.exists():
            logger.error(f"Model '{args.model_name}.onnx' not found in {models_dir}")
            logger.error(f"\nAvailable models:")
            for model in onnx_models:
                logger.error(f"  - {model.stem}")
            exit(1)
    else:
        model_path = onnx_models[0]
        logger.info(f"No model specified, using: {model_path.stem}\n")

    analyzer = load_and_analyze_onnx_model(model_path)
    analyzer._build_tensor_info()

    if analyzer:
        logger.info(f"\nExtracted {len(analyzer.weights)} weight tensors")
        logger.info(f"Found {len(analyzer.layers)} layers")

        for i, layer in enumerate(analyzer.layers):
            logger.info(f"\nLayer {i+1}:")
            logger.info(f"  Name: {layer['name']}")
            logger.info(f"  Type: {layer['op_type']}")
            logger.info(f"  Inputs: {layer['inputs']}")
            logger.info(f"  Outputs: {layer['outputs']}")
            logger.info(f"  Attributes: {layer['attributes']}")
