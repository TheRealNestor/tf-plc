"""
Type descriptions related to the neural network.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ActivationType(Enum):
    """Types of activation functions supported"""

    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"


@dataclass(frozen=True)
class ActivationLayer:
    """Represents an activation function layer"""

    layer_id: int
    activation: ActivationType
    input_size: int
    output_size: int


@dataclass(frozen=True)
class MatMulLayer:
    """Represents an ONNX MatMul layer"""

    layer_id: int
    weights: np.ndarray
    input_size: int
    output_size: int

@dataclass(frozen=True)
class AddLayer:
    """Represents an ONNX Add layer"""

    layer_id: int
    bias: np.ndarray
    input_size: int
    output_size: int


@dataclass(frozen=True)
class GemmLayer:
    """Represents an ONNX Gemm layer"""

    layer_id: int
    weights: np.ndarray
    bias: Optional[np.ndarray]
    input_size: int
    output_size: int

    alpha: float = 1.0
    beta: float = 1.0
    transA : bool = False
    transB : bool = False

@dataclass(frozen=True)
class FusedGemmLayer:
    """Represents a fused Gemm + Activation layer"""

    layer_id: int
    weights: np.ndarray
    bias: Optional[np.ndarray]
    activation: ActivationType
    input_size: int
    output_size: int

    alpha: float = 1.0
    beta: float = 1.0
    transA : bool = False
    transB : bool = False


@dataclass(frozen=True)
class QuantizeLinearLayer:
    """Represents an ONNX QuantizeLinear layer"""

    layer_id: int
    input_name: str
    scale_name: str
    zero_point_name: str
    output_name: str
    axis: Optional[int] = None

@dataclass(frozen=True)
class DequantizeLinearLayer:
    """Represents an ONNX DequantizeLinear layer"""

    layer_id: int
    input_name: str
    scale_name: str
    zero_point_name: str
    output_name: str
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
