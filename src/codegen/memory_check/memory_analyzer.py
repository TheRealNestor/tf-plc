"""
Memory analyzer for PLC deployment.
Computes memory requirements and validates against device limits.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from ..ir_to_st.type_conversion import get_type_size_bytes

from ..types import (
    NetworkIR,
    BaseLayer,
    MatMulLayer,
    GemmLayer,
    FusedGemmLayer,
    FusedLinearLayer,
    AddLayer,
    QuantizeLinearLayer,
    DequantizeLinearLayer,
)

logger = logging.getLogger(__name__)

# Map dtype strings to sizes in bytes
DTYPE_SIZES: Dict[str, int] = {
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}

DEFAULT_ELEMENT_SIZE = 4  # REAL (float32)

# TODO: Make elegant and unified way of handling dtype sizes for the different formats


@dataclass
class MemoryBreakdown:
    """Detailed memory usage breakdown."""

    weights_bytes: int = 0
    biases_bytes: int = 0
    activations_bytes: int = 0
    constants_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return (
            self.weights_bytes
            + self.biases_bytes
            + self.activations_bytes
            + self.constants_bytes
        )

    @property
    def total_kb(self) -> float:
        return self.total_bytes / 1024

    def __str__(self) -> str:
        return (
            f"Memory Breakdown:\n"
            f"  Weights:     {self.weights_bytes:>8} bytes ({self.weights_bytes/1024:.2f} KB)\n"
            f"  Biases:      {self.biases_bytes:>8} bytes ({self.biases_bytes/1024:.2f} KB)\n"
            f"  Activations: {self.activations_bytes:>8} bytes ({self.activations_bytes/1024:.2f} KB)\n"
            f"  Constants:   {self.constants_bytes:>8} bytes ({self.constants_bytes/1024:.2f} KB)\n"
            f"  ─────────────────────────────\n"
            f"  TOTAL:       {self.total_bytes:>8} bytes ({self.total_kb:.2f} KB)"
        )


@dataclass
class MemoryCheckResult:
    """Result of memory validation check."""

    fits_in_memory: bool
    breakdown: MemoryBreakdown
    limit_bytes: int
    utilization_percent: float
    warnings: List[str]
    errors: List[str]

    def __str__(self) -> str:
        status = "✓ PASS" if self.fits_in_memory else "✗ FAIL"
        limit_kb = self.limit_bytes / 1024
        return (
            f"\nMemory Check: {status}\n"
            f"{'=' * 40}\n"
            f"{self.breakdown}\n"
            f"{'=' * 40}\n"
            f"Device Limit: {limit_kb:.2f} KB\n"
            f"Utilization:  {self.utilization_percent:.1f}%\n"
            f"Remaining:    {(self.limit_bytes - self.breakdown.total_bytes)/1024:.2f} KB"
        )


def _get_element_size(dtype: Optional[str]) -> int:
    """Get element size in bytes from dtype string."""
    if not dtype:
        return DEFAULT_ELEMENT_SIZE

    return get_type_size_bytes(dtype)


def _compute_layer_weights(layer: BaseLayer) -> Tuple[int, int]:
    """Compute weight and bias memory for a layer in bytes."""
    weights_bytes = 0
    biases_bytes = 0

    # Linear layers (MatMul, Gemm, Fused variants)
    if isinstance(layer, (MatMulLayer, GemmLayer, FusedGemmLayer, FusedLinearLayer)):
        # Get weight dtype from layer if available
        weight_dtype = getattr(layer, "weight_type", None) or layer.input_type
        weight_element_size = _get_element_size(weight_dtype)

        # Weight matrix: input_size x output_size
        weight_elements = layer.input_size * layer.output_size
        weights_bytes = weight_elements * weight_element_size

        # Bias dtype (often float even for quantized weights)
        bias_dtype = getattr(layer, "bias_type", None) or layer.output_type
        bias_element_size = _get_element_size(bias_dtype)

        # Determine if layer has bias
        # Check multiple ways since different layer types store this differently
        has_bias = False

        # Method 1: Explicit has_bias attribute
        if hasattr(layer, "has_bias"):
            has_bias = layer.has_bias
        # Method 2: bias attribute is not None
        elif hasattr(layer, "bias") and layer.bias is not None:
            has_bias = True
        # Method 3: bias_name attribute exists and is not empty
        elif hasattr(layer, "bias_name") and layer.bias_name:
            has_bias = True
        # Method 4: Gemm layers typically always have bias
        elif isinstance(layer, (GemmLayer, FusedGemmLayer)):
            has_bias = True
        # Method 5: FusedLinearLayer with activation implies bias was present
        elif isinstance(layer, FusedLinearLayer):
            # Check if there's a bias input in the layer's inputs
            # Typically: input, weight, bias
            has_bias = len(layer.inputs) >= 3 or getattr(layer, "has_bias", True)

        if has_bias:
            biases_bytes = layer.output_size * bias_element_size

        logger.debug(
            f"Layer {layer.name}: weights={weight_elements} ({weights_bytes}B), "
            f"has_bias={has_bias}, bias_size={biases_bytes}B"
        )

    # Standalone Add layer used as bias
    elif isinstance(layer, AddLayer):
        if getattr(layer, "is_bias", False):
            element_size = _get_element_size(layer.output_type)
            biases_bytes = layer.output_size * element_size

    return weights_bytes, biases_bytes


def _compute_activation_bytes(layer: BaseLayer) -> Tuple[int, int]:
    """Compute input and output activation sizes in bytes."""
    input_size = _get_element_size(layer.input_type) * layer.input_size
    output_size = _get_element_size(layer.output_type) * layer.output_size
    return input_size, output_size


def _compute_constants_size(layer: BaseLayer) -> int:
    """Compute size of constants (quantization params, etc.)."""
    if isinstance(layer, QuantizeLinearLayer):
        scale_type = getattr(layer, "scale_type", "float32")
        scale_size = _get_element_size(scale_type)
        zp_size = _get_element_size(layer.output_type)
        return scale_size + zp_size

    elif isinstance(layer, DequantizeLinearLayer):
        scale_type = getattr(layer, "scale_type", "float32")
        scale_size = _get_element_size(scale_type)
        zp_size = _get_element_size(layer.input_type)
        return scale_size + zp_size

    return 0


def _estimate_activation_memory(
    ir: NetworkIR, buffer_allocations: Optional[Dict[str, str]] = None
) -> int:
    """
    Estimate activation memory needed.

    If buffer_allocations provided, use actual buffer reuse.
    Otherwise, assume separate output variable for each layer.

    Args:
        ir: Network IR
        buffer_allocations: Optional dict mapping tensor names to buffer names

    Returns:
        Activation memory in bytes
    """
    if buffer_allocations:
        # Calculate actual buffer sizes from allocations
        buffer_sizes: Dict[str, int] = {}

        for layer_name in ir.execution_order:
            layer = ir.get_layer(layer_name)

            for output_tensor in layer.outputs:
                # Skip network outputs (they need their own storage)
                if output_tensor in ir.output_tensors:
                    continue

                buffer_name = buffer_allocations.get(output_tensor)
                if buffer_name:
                    tensor_bytes = (
                        _get_element_size(layer.output_type) * layer.output_size
                    )
                    buffer_sizes[buffer_name] = max(
                        buffer_sizes.get(buffer_name, 0), tensor_bytes
                    )

        # Also account for network inputs/outputs (not in buffer pool)
        io_bytes = 0
        for layer in ir.layers.values():
            for inp in layer.inputs:
                if inp in ir.input_tensors:
                    io_bytes = max(
                        io_bytes, _get_element_size(layer.input_type) * layer.input_size
                    )
            for out in layer.outputs:
                if out in ir.output_tensors:
                    io_bytes = max(
                        io_bytes,
                        _get_element_size(layer.output_type) * layer.output_size,
                    )

        total_buffer_bytes = sum(buffer_sizes.values())
        logger.info(
            f"Activation memory (buffer allocation): {len(buffer_sizes)} buffers = {total_buffer_bytes} bytes, "
            f"I/O = {io_bytes} bytes, total = {total_buffer_bytes + io_bytes} bytes"
        )
        return total_buffer_bytes + io_bytes
    else:
        # No buffer allocation: each layer gets its own output variable
        # Sum all intermediate tensor sizes
        total_activation_bytes = 0

        for layer in ir.layers.values():
            # Each layer's output needs storage
            output_bytes = _get_element_size(layer.output_type) * layer.output_size
            total_activation_bytes += output_bytes

        # Network input also needs storage
        input_bytes = 0
        for layer in ir.layers.values():
            for inp in layer.inputs:
                if inp in ir.input_tensors:
                    input_bytes = max(
                        input_bytes,
                        _get_element_size(layer.input_type) * layer.input_size,
                    )
                    break
            if input_bytes > 0:
                break

        total_activation_bytes += input_bytes

        logger.info(
            f"Activation memory (no buffer reuse): {len(ir.layers)} layer outputs + input = "
            f"{total_activation_bytes} bytes"
        )
        return total_activation_bytes


def analyze_memory(
    ir: NetworkIR,
    memory_limit_bytes: int = 96 * 1024,
    buffer_allocations: Optional[Dict[str, str]] = None,
) -> MemoryCheckResult:
    """
    Analyze memory requirements for the network IR.

    Args:
        ir: Network IR (should be after optimization passes)
        memory_limit_bytes: Device memory limit in bytes (default: 96 KB)
        buffer_allocations: Optional buffer allocations from optimizer

    Returns:
        MemoryCheckResult with detailed breakdown
    """
    breakdown = MemoryBreakdown()
    warnings: List[str] = []
    errors: List[str] = []

    for layer in ir.layers.values():
        layer_weights, layer_biases = _compute_layer_weights(layer)
        breakdown.weights_bytes += layer_weights
        breakdown.biases_bytes += layer_biases
        breakdown.constants_bytes += _compute_constants_size(layer)

    breakdown.activations_bytes = _estimate_activation_memory(ir, buffer_allocations)

    # Validation
    total = breakdown.total_bytes
    utilization = (total / memory_limit_bytes) * 100
    fits = total <= memory_limit_bytes

    # Generate warnings
    if utilization > 80:
        warnings.append(
            f"Memory utilization is high ({utilization:.1f}%). "
            f"Consider quantization or model pruning."
        )

    if breakdown.weights_bytes > memory_limit_bytes * 0.7:
        warnings.append(
            f"Weights use {breakdown.weights_bytes/1024:.1f} KB "
            f"({breakdown.weights_bytes/memory_limit_bytes*100:.1f}% of limit). "
            f"Consider reducing model size."
        )

    # Generate errors
    if not fits:
        overflow = total - memory_limit_bytes
        errors.append(
            f"Model exceeds memory limit by {overflow/1024:.2f} KB. "
            f"Required: {total/1024:.2f} KB, Limit: {memory_limit_bytes/1024:.2f} KB"
        )

    return MemoryCheckResult(
        fits_in_memory=fits,
        breakdown=breakdown,
        limit_bytes=memory_limit_bytes,
        utilization_percent=utilization,
        warnings=warnings,
        errors=errors,
    )


def check_memory(
    ir: NetworkIR,
    memory_limit_kb: float = 96,
    fail_on_exceed: bool = True,
    buffer_allocations: Optional[Dict[str, str]] = None,
) -> MemoryCheckResult:
    """
    Check memory requirements and optionally fail if exceeded.

    Args:
        ir: Network IR (post-optimization)
        memory_limit_kb: Memory limit in KB (default: 96)
        fail_on_exceed: Raise exception if memory exceeded
        buffer_allocations: Optional buffer allocations from optimizer

    Returns:
        MemoryCheckResult

    Raises:
        MemoryError: If fail_on_exceed=True and model exceeds limit
    """
    result = analyze_memory(
        ir,
        memory_limit_bytes=int(memory_limit_kb * 1024),
        buffer_allocations=buffer_allocations,
    )

    logger.info(str(result))

    for warning in result.warnings:
        logger.warning(warning)

    if result.errors:
        for error in result.errors:
            logger.error(error)

        if fail_on_exceed:
            raise MemoryError(
                f"Model exceeds PLC memory limit. "
                f"Required: {result.breakdown.total_kb:.2f} KB, "
                f"Limit: {memory_limit_kb:.2f} KB"
            )

    return result
