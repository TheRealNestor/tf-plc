"""
Type descriptions related to the neural network.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
from enum import Enum


class ActivationType(Enum):
    """Types of activation functions supported"""

    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"


@dataclass(frozen=True, kw_only=True)
class BaseLayer:
    """Base class for all layers"""

    layer_id: int
    name: str
    op_type: str
    input_size: int
    output_size: int
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()

    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    input_type: Optional[str] = None
    output_type: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class ActivationLayer(BaseLayer):
    """Represents an activation function layer"""

    activation: ActivationType


@dataclass(frozen=True, kw_only=True)
class ConstFoldOptLayer(BaseLayer):
    """Represents a constant folding optimization layer"""

    folded_tensor: np.ndarray


@dataclass(frozen=True, kw_only=True)
class LinearLayer(BaseLayer):
    """Base class for layers with weights and biases"""

    weights: np.ndarray
    bias: Optional[np.ndarray] = None  # Some linear layers may have bias


@dataclass(frozen=True, kw_only=True)
class MatMulLayer(LinearLayer):
    """Base class for MatMul layers"""

    pass


@dataclass(frozen=True, kw_only=True)
class GemmLayer(LinearLayer):
    """Base class for Gemm layers"""

    alpha: float = 1.0
    beta: float = 1.0
    transA: bool = False
    transB: bool = False


@dataclass(frozen=True, kw_only=True)
class FusedGemmLayer(GemmLayer):
    """Base class for Fused Gemm + Activation layers"""

    activation: ActivationType


@dataclass(frozen=True, kw_only=True)
class AddLayer(BaseLayer):
    """Represents an ONNX Add layer"""
    bias: np.ndarray


@dataclass(frozen=True, kw_only=True)
class ReshapeLayer(BaseLayer):
    """Represents an ONNX Reshape layer"""

    # I think we have enough information from baselayer to define the reshape layer
    pass


@dataclass(frozen=True, kw_only=True)
class QuantizeLinearLayer(BaseLayer):
    """Represents an ONNX QuantizeLinear layer"""

    scale_name: str
    zero_point_name: str
    axis: Optional[int] = None


@dataclass(frozen=True, kw_only=True)
class DequantizeLinearLayer(BaseLayer):
    """Represents an ONNX DequantizeLinear layer"""

    scale_name: str
    zero_point_name: str
    axis: Optional[int] = None

@dataclass(frozen=True)
class NetworkIR:
    """Intermediate graph-based representation of a neural network"""

    # layer_name -> layer
    layers: Dict[str, BaseLayer]

    # List of layer names in execution order (topological sort)
    execution_order: List[str]

    # tensor_name -> layer_name
    tensor_producers: Dict[str, str] = field(default_factory=dict)

    # tensor_name -> [layer_names]
    tensor_consumers: Dict[str, List[str]] = field(default_factory=dict)

    input_tensors: Tuple[str, ...] = ()
    output_tensors: Tuple[str, ...] = ()

    def get_layer(self, name: str) -> BaseLayer:
        """Get layer by name"""
        return self.layers[name]

    def get_input_layers(self, layer_name: str) -> List[str]:
        """Get names of layers that produce inputs for the given layer"""
        return [
            self.tensor_producers[tensor_name]
            for tensor_name in self.get_layer(layer_name).inputs
            if tensor_name in self.tensor_producers
        ]
    
    def get_output_layers(self, layer_name: str) -> List[str]:
        """Get names of layers that consume outputs from the given layer"""
        return [    
            self.tensor_consumers[tensor_name]
            for tensor_name in self.get_layer(layer_name).outputs
            if tensor_name in self.tensor_consumers
        ]
    
    def is_network_input(self, tensor_name: str) -> bool:
        """Check if a tensor is a network input"""
        return tensor_name in self.input_tensors
    
    def is_network_output(self, tensor_name: str) -> bool:
        """Check if a tensor is a network output"""
        return tensor_name in self.output_tensors
    
    def __str__(self) -> str:
        layer_types = [type(layer).__name__ for layer in self.layers.values()]
        layers_str = "\n  ".join(layer_types)
        return (
            f"NetworkIR(layers={len(self.layers)})\n"
            f"Layer types (in order):\n  {layers_str}"
        )
