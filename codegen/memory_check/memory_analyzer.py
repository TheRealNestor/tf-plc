"""
Memory analyzer for PLC deployment.
Computes memory requirements and validates against device limits.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

    dtype_lower = dtype.lower()

    if dtype_lower in DTYPE_SIZES:
        return DTYPE_SIZES[dtype_lower]

    # Partial match (e.g., "numpy.float32" -> "float32")
    for known_dtype, size in DTYPE_SIZES.items():
        if known_dtype in dtype_lower:
            return size

    logger.warning(
        f"Unknown dtype '{dtype}', using default size {DEFAULT_ELEMENT_SIZE}"
    )
    return DEFAULT_ELEMENT_SIZE


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


def _estimate_activation_memory(ir: NetworkIR) -> int:
    """
    Estimate activation memory needed.
    """
    max_input_bytes = 0
    max_output_bytes = 0

    for layer in ir.layers.values():
        input_bytes, output_bytes = _compute_activation_bytes(layer)
        max_input_bytes = max(max_input_bytes, input_bytes)
        max_output_bytes = max(max_output_bytes, output_bytes)

    return max_input_bytes + max_output_bytes


def analyze_memory(
    ir: NetworkIR, memory_limit_bytes: int = 96 * 1024
) -> MemoryCheckResult:
    """
    Analyze memory requirements for the network IR.

    Args:
        ir: Network IR (should be after optimization passes)
        memory_limit_bytes: Device memory limit in bytes (default: 96 KB)

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

    breakdown.activations_bytes = _estimate_activation_memory(ir)

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


def get_layer_memory_report(ir: NetworkIR) -> str:
    """Generate a per-layer memory report."""
    lines = ["Per-Layer Memory Report", "=" * 80]
    lines.append(
        f"{'Layer':30} | {'Weights':>10} | {'Biases':>10} | "
        f"{'In':>6} | {'Out':>6} | {'dtype'}"
    )
    lines.append("-" * 80)

    total_weights = 0
    total_biases = 0

    for name, layer in ir.layers.items():
        weights, biases = _compute_layer_weights(layer)
        total_weights += weights
        total_biases += biases

        if weights > 0 or biases > 0:
            weight_dtype = getattr(layer, "weight_type", layer.input_type) or "?"
            lines.append(
                f"{name:30} | "
                f"{weights:>7} B | "
                f"{biases:>7} B | "
                f"{layer.input_size:>6} | "
                f"{layer.output_size:>6} | "
                f"{weight_dtype}"
            )

    lines.append("=" * 80)
    lines.append(f"{'TOTAL':30} | {total_weights:>7} B | {total_biases:>7} B |")

    return "\n".join(lines)


def check_memory(
    ir: NetworkIR,
    memory_limit_kb: float = 96,
    fail_on_exceed: bool = True,
) -> MemoryCheckResult:
    """
    Check memory requirements and optionally fail if exceeded.

    Args:
        ir: Network IR (post-optimization)
        memory_limit_kb: Memory limit in KB (default: 96)
        fail_on_exceed: Raise exception if memory exceeded

    Returns:
        MemoryCheckResult

    Raises:
        MemoryError: If fail_on_exceed=True and model exceeds limit
    """
    result = analyze_memory(ir, memory_limit_bytes=int(memory_limit_kb * 1024))

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
