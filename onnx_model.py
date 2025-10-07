# Translate onnx model to PLC Structured Text (IEC 61131-3)

# Start with simple feedforward network with relu activations

import onnx
import numpy as np
from typing import Dict, List, Tuple, Any
import os

class ONNXModel:
    """
    A class to load and analyze ONNX models, extracting weights, layer information,
    and model structure for later code generation.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the analyzer with an ONNX model.
        
        Args:
            model_path (str): Path to the ONNX model file
        """
        self.model_path = model_path
        self.model = None
        self.graph = None
        self.weights = {}
        self.layers = []
        self.input_info = {}
        self.output_info = {}
        
    def load_model(self) -> bool:
        """
        Load the ONNX model from file.
        
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"Error: Model file {self.model_path} not found")
                return False
                
            self.model = onnx.load(self.model_path)
            
            onnx.checker.check_model(self.model)
            print(f"Successfully loaded ONNX model: {self.model_path}")
            
            self.graph = self.model.graph
            return True
            
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            return False
    
    def extract_weights(self) -> Dict[str, np.ndarray]:
        """
        Extract all weights and biases from the model.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping parameter names to numpy arrays
        """
        if not self.model:
            print("Model not loaded. Call load_model() first.")
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
            print("Model not loaded. Call load_model() first.")
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
            print("Model not loaded. Call load_model() first.")
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


def load_and_analyze_onnx_model(model_path: str) -> ONNXModel:
    """
    Convenience function to load and analyze an ONNX model.
    
    Args:
        model_path (str): Path to the ONNX model file
        
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
    models_dir = "models"
    onnx_models = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]

    if not onnx_models:
        raise FileNotFoundError("No ONNX models found in the models directory")    

    onxx_model = onnx_models[0]
    model_path = os.path.join(models_dir, onxx_model)

    print(f"Found ONNX model: {model_path}")
    
    analyzer = load_and_analyze_onnx_model(model_path)
    
    if analyzer:
        weights = analyzer.weights
        layers = analyzer.layers
        
        print(f"\nExtracted {len(weights)} weight tensors")
        print(f"Found {len(layers)} layers")
        
        print("\nFirst 3 layers in detail:")
        for i, layer in enumerate(layers[:3]):
            print(f"\nLayer {i+1}:")
            print(f"  Name: {layer['name']}")
            print(f"  Type: {layer['op_type']}")
            print(f"  Inputs: {layer['inputs']}")
            print(f"  Outputs: {layer['outputs']}")
            print(f"  Attributes: {layer['attributes']}")


