"""
Type descriptions related to the neural network.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ActivationType(Enum):
    """Types of activation functions supported"""

    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"

@dataclass(frozen=True)
class BaseLayer:
    """Base class for all layers"""

    layer_id: int
    name: str
    op_type: str
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    attributes: dict = field(default_factory=dict)

    input_size: int
    output_size: int
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    input_type: Optional[str] = None
    output_type: Optional[str] = None

@dataclass(frozen=True)
class ActivationLayer(BaseLayer):
    """Represents an activation function layer"""
    activation: ActivationType


@dataclass(frozen=True)
class LinearLayer(BaseLayer):
    """Base class for layers with weights and biases"""

    weights: np.ndarray
    bias: Optional[np.ndarray] = None # Some linear layers may have bias

@dataclass(frozen=True)
class MatMulLayer(LinearLayer):
    """Base class for MatMul layers"""
    pass

@dataclass(frozen=True)
class GemmLayer(LinearLayer):
    """Base class for Gemm layers"""
    alpha: float = 1.0
    beta: float = 1.0
    transA : bool = False
    transB : bool = False

@dataclass(frozen=True)
class FusedGemmLayer(GemmLayer):
    """Base class for Fused Gemm + Activation layers"""
    activation: ActivationType


@dataclass(frozen=True)
class AddLayer(BaseLayer):
    """Represents an ONNX Add layer"""
    bias: np.ndarray


@dataclass(frozen=True)
class ReshapeLayer(BaseLayer):
    """Represents an ONNX Reshape layer"""
    # TODO: I think I need to add additional information here...
    pass

@dataclass(frozen=True)
class QuantizeLinearLayer(BaseLayer):
    """Represents an ONNX QuantizeLinear layer"""

    scale_name: str
    zero_point_name: str
    axis: Optional[int] = None

@dataclass(frozen=True)
class DequantizeLinearLayer(BaseLayer):
    """Represents an ONNX DequantizeLinear layer"""

    scale_name: str
    zero_point_name: str
    axis: Optional[int] = None

@dataclass(frozen=True)
class NetworkIR:
    """Intermediate representation of the neural network"""

    input_size: int
    output_size: int
    layers: Tuple[object, ...]

    def __str__(self) -> str:
        layer_types = [type(layer).__name__ for layer in self.layers]
        layers_str = "\n  ".join(layer_types)
        return (
            f"NetworkIR(input={self.input_size}, output={self.output_size}, layers={len(self.layers)})\n"
            f"Layer types (in order):\n  {layers_str}"
        )
