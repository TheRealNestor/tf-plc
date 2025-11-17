"""
Loads and analyzes ONNX models to extract weights, layer information, and model structure for code generation.
"""

import onnx
import numpy as np
from typing import Dict, List, Tuple, Any
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
        self.tensor_info = {} # Maps tensor names to their types and shapes

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

    def _build_tensor_info(self):
        """Build comprehensive tensor information from ONNX model by using ONNX inference engine."""
        if not self.model:
            return

        try:
            logger.debug("Running ONNX shape inference...")
            inferred_model = onnx.shape_inference.infer_shapes(self.model)
            # Update our model with the inferred one
            self.model = inferred_model
            self.graph = self.model.graph
            logger.debug(f"Shape inference complete. value_info entries: {len(self.graph.value_info)}")
        except Exception as e:
            logger.warning(f"Shape inference failed: {e}, continuing with original model.")

        tensor_info = {}

        # Process model inputs
        for input_tensor in self.graph.input:
            tensor_info[input_tensor.name] = {
                'dtype': self.get_tensor_type_name(input_tensor.type.tensor_type.elem_type),
                'shape': [dim.dim_value if dim.dim_value > 0 else dim.dim_param if dim.dim_param else -1 
                        for dim in input_tensor.type.tensor_type.shape.dim]
            }

        # Process model outputs
        for output_tensor in self.graph.output:
            tensor_info[output_tensor.name] = {
                'dtype': self.get_tensor_type_name(output_tensor.type.tensor_type.elem_type),
                'shape': [dim.dim_value if dim.dim_value > 0 else dim.dim_param if dim.dim_param else -1 
                        for dim in output_tensor.type.tensor_type.shape.dim]
            }

        # Process intermediate value_info (NOW POPULATED BY SHAPE INFERENCE!)
        for value_info in self.graph.value_info:
            tensor_info[value_info.name] = {
                'dtype': self.get_tensor_type_name(value_info.type.tensor_type.elem_type),
                'shape': [dim.dim_value if dim.dim_value > 0 else dim.dim_param if dim.dim_param else -1 
                        for dim in value_info.type.tensor_type.shape.dim]
            }

        # Process initializers/weights (they also have type info)
        for init in self.graph.initializer:
            tensor_info[init.name] = {
                'dtype': self.get_tensor_type_name(init.data_type),
                'shape': list(init.dims)
            }

        self.tensor_info = tensor_info

        logger.info(f"Built tensor_info with {len(tensor_info)} tensors")

        if logger.isEnabledFor(logging.DEBUG):
            for name, info in list(tensor_info.items())[:15]:
                logger.debug(f"  {name}: dtype={info['dtype']}, shape={info['shape']}")
            if len(tensor_info) > 15:
                logger.debug(f"  ... and {len(tensor_info) - 15} more")

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

        layers = []

        for node in self.graph.node:
            layer_info = {
                'name': node.name,
                'op_type': node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
                'attributes': {}
            }

            # Extract attributes
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    layer_info['attributes'][attr.name] = attr.i
                elif attr.type == onnx.AttributeProto.FLOAT:
                    layer_info['attributes'][attr.name] = attr.f
                elif attr.type == onnx.AttributeProto.STRING:
                    layer_info['attributes'][attr.name] = attr.s.decode('utf-8')
                elif attr.type == onnx.AttributeProto.INTS:
                    layer_info['attributes'][attr.name] = list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    layer_info['attributes'][attr.name] = list(attr.floats)

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
                    'shape': shape,
                    'dtype': input_tensor.type.tensor_type.elem_type
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
                'shape': shape,
                'dtype': output_tensor.type.tensor_type.elem_type
            }

        self.input_info = input_info
        self.output_info = output_info
        return input_info, output_info

    def print_model_summary(self):
        """Print a comprehensive summary of the model."""
        if not self.model:
            print("Model not loaded. Call load_model() first.")
            return

        print("\n" + "="*60)
        print("ONNX MODEL SUMMARY")
        print("="*60)

        print(f"Model path: {self.model_path}")
        print(f"IR Version: {self.model.ir_version}")
        print(f"Producer: {self.model.producer_name} {self.model.producer_version}")

        input_info, output_info = self.get_input_output_info()
        print(f"\nInputs ({len(input_info)}):")
        for name, info in input_info.items():
            print(f"  - {name}: shape={info['shape']}, dtype={info['dtype']}")

        print(f"\nOutputs ({len(output_info)}):")
        for name, info in output_info.items():
            print(f"  - {name}: shape={info['shape']}, dtype={info['dtype']}")

        layers = self.analyze_layers()
        print(f"\nLayers ({len(layers)}):")
        layer_types = {}
        for layer in layers:
            op_type = layer['op_type']
            layer_types[op_type] = layer_types.get(op_type, 0) + 1
            print(f"  - {layer['name']}: {op_type}")

        print(f"\nLayer type counts:")
        for op_type, count in layer_types.items():
            print(f"  - {op_type}: {count}")

        weights = self.extract_weights()
        print(f"\nWeights/Parameters ({len(weights)}):")
        total_params = 0
        for name, weight in weights.items():
            param_count = np.prod(weight.shape)
            total_params += param_count
            print(f"  - {name}: shape={weight.shape}, params={param_count}")

        print(f"\nTotal parameters: {total_params:,}")
        print("="*60)


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
        analyzer.print_model_summary()
        return analyzer
    else:
        return None


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load and analyze ONNX model')
    parser.add_argument('model_name', nargs='?', help='Name of the ONNX model file (without .onnx extension)')
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s"
    )    

    models_dir = Path("models") / "onnx"
    
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
