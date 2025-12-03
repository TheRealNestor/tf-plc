"""
Fuse MatMul → Add (Bias) → Activation into FusedLinearLayer.

This optimization reduces memory usage by eliminating intermediate arrays
and improves performance by combining operations into a single loop.
"""

import logging
from typing import Optional

from ..base_pass import OptimizationPass
from ...types import (
    NetworkIR,
    MatMulLayer,
    AddLayer,
    ActivationLayer,
    FusedLinearLayer,
)

logger = logging.getLogger(__name__)


class FuseLinearActivationPass(OptimizationPass):
    """Fuse MatMul → Add (Bias) → Activation chains into FusedLinearLayer."""

    def get_name(self) -> str:
        return "fuse_linear_activation"

    def optimize(self, network: NetworkIR) -> None:
        """Find and fuse MatMul → Add → Activation chains."""

        fused_count = 0

        for layer_name in list(network.execution_order):
            layer = network.get_layer(layer_name)

            # Look for MatMul layers
            if isinstance(layer, MatMulLayer):
                chain = self._find_fuseable_chain(network, layer_name)

                if chain:
                    matmul, add_layer, activation = chain
                    bias = add_layer.bias if hasattr(add_layer, "bias") else None

                    if bias is not None:
                        fused_layer = self._create_fused_layer(
                            matmul, add_layer, activation, bias
                        )

                        # Replace matmul with fused layer (keeps same name/slot)
                        self.replace_layer(matmul, fused_layer, network)

                        # Remap intermediate outputs to fused layer output
                        for output_tensor in add_layer.outputs:
                            self.remap_tensor(output_tensor, fused_layer.outputs[0])

                        # Remove the other layers in the chain
                        self.mark_for_removal(add_layer)
                        self.mark_for_removal(activation)

                        fused_count += 1
                        logger.debug(
                            f"Fused layers: {matmul.name} + {add_layer.name} + {activation.name} "
                            f"into {fused_layer.name}"
                        )

        if fused_count > 0:
            logger.info(f"Fused {fused_count} linear layer chains")

    def _find_fuseable_chain(
        self, network: NetworkIR, matmul_name: str
    ) -> Optional[tuple[MatMulLayer, AddLayer, ActivationLayer]]:
        """
        Find MatMul → Add → Activation chain.

        Returns:
            (MatMul, Add, Activation) tuple if fuseable, None otherwise
        """
        matmul = network.get_layer(matmul_name)

        # MatMul must have exactly one consumer
        consumers = network.get_output_layers(matmul_name)
        if len(consumers) != 1:
            return None

        # Next layer must be Add (bias)
        add_layer = network.get_layer(consumers[0])
        if not isinstance(add_layer, AddLayer):
            return None

        # Add must have exactly one consumer
        add_consumers = network.get_output_layers(add_layer.name)
        if len(add_consumers) != 1:
            return None

        # Next layer must be Activation
        activation = network.get_layer(add_consumers[0])
        if not isinstance(activation, ActivationLayer):
            return None

        return (matmul, add_layer, activation)

    def _create_fused_layer(
        self,
        matmul: MatMulLayer,
        add_layer: AddLayer,
        activation: ActivationLayer,
        bias,
    ) -> FusedLinearLayer:
        """Create a fused linear layer from the chain components."""
        return FusedLinearLayer(
            name=f"{matmul.name}/Fused",
            layer_id=matmul.layer_id,
            op_type="FusedLinear",
            inputs=matmul.inputs,
            outputs=activation.outputs,
            input_size=matmul.input_size,
            output_size=matmul.output_size,
            input_shape=matmul.input_shape,
            output_shape=activation.output_shape,
            input_type=matmul.input_type,
            output_type=activation.output_type,
            weights=matmul.weights,
            weight_scale=matmul.weight_scale,
            weight_zero_point=matmul.weight_zero_point,
            bias=bias,
            activation=activation.activation,
        )
