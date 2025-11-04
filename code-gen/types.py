"""
Type descriptions related to the neural network.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ActivationType(Enum):
    """Types of activation functions supported"""

    NONE = "none"
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"


@dataclass(frozen=True)
class DenseLayer:
    """Represents an ONNX dense/fully-connected layer"""

    layer_id: int
    weights: np.ndarray
    bias: Optional[np.ndarray]
    activation: ActivationType
    input_size: int
    output_size: int

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
    layers: Tuple[DenseLayer, ...]

    def __str__(self) -> str:
        return f"NetworkIR(input={self.input_size}, output={self.output_size}, layers={len(self.layers)})"
