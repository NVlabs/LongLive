from typing import Any

import torch
from fouroversix.matmul import fp4_matmul
from fouroversix.model.config import ModuleQuantizationConfig
from fouroversix.model.quantize import QuantizedModule
from fouroversix.quantize import (
    QuantizationConfig,
    QuantizedTensor,
    quantize_to_fp4,
)
from torch import nn
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssExperts,
    GptOssMLP,
    GptOssTopKRouter,
)


@QuantizedModule.register(GptOssMLP)
class FourOverSixGptOssMLP(nn.Module):
    """Drop-in replacement for GptOssMLP layer that uses FP4 quantization."""

    def __init__(
        self,
        module: GptOssMLP,
        config: ModuleQuantizationConfig,
    ) -> None:
        """
        Initialize the FourOverSixGptOssMLP layer.

        Args:
            module (GptOssMLP): The high-precision module that this quantized layer will
                replace.
            config (ModuleQuantizationConfig): The quantization configuration to use for
                the layer.

        """

        super().__init__()

        self.config = config

        gpt_oss_config = GptOssConfig(
            num_local_experts=module.experts.num_experts,
            hidden_size=module.experts.hidden_size,
            intermediate_size=module.experts.intermediate_size,
            num_experts_per_token=module.router.top_k,
        )

        self.router = GptOssTopKRouter(gpt_oss_config)
        self.router.weight = module.router.weight
        self.router.bias = module.router.bias

        self.experts = FourOverSixGptOssExperts(
            module.experts,
            quantization_config=self.config,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the FP4 MLP layer."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        _, router_scores, router_indices = self.router(hidden_states)
        hidden_states = self.experts(hidden_states, router_indices, router_scores)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_scores

    @property
    def parameters_to_quantize(self) -> tuple[str, ...]:
        """Return high precision parameters to be quantized and deleted."""
        return ()

    def get_element_size(self, parameter_name: str) -> float:
        """Get the size of a single element, in bytes, for a parameter."""

        return {
            "quantized_down_proj_values": 0.5,
            "quantized_gate_up_proj_values": 0.5,
            "down_proj": 9 / 16,
            "gate_up_proj": 9 / 16,
        }.get(
            parameter_name,
            getattr(self, parameter_name).element_size(),
        )


@QuantizedModule.register(GptOssExperts, replace_existing_modules_in_model=False)
class FourOverSixGptOssExperts(nn.Module):
    """Drop-in replacement for GptOssExperts layer that uses FP4 quantization."""

    def __init__(
        self,
        module: GptOssExperts,
        quantization_config: ModuleQuantizationConfig | None = None,
    ) -> None:

        super().__init__()

        self.num_experts = module.num_experts
        self.intermediate_size = module.intermediate_size
        self.hidden_size = module.hidden_size

        self.down_proj_bias = module.down_proj_bias
        self.gate_up_proj_bias = module.gate_up_proj_bias

        # Store original weights so they can be quantized and then deleted
        self.down_proj = module.down_proj
        self.gate_up_proj = module.gate_up_proj

        self.config = quantization_config

        if not self.config.keep_master_weights:
            self.register_buffer(
                "quantized_down_proj_values",
                nn.Parameter(
                    torch.zeros(
                        self.num_experts,
                        self.intermediate_size,
                        self.hidden_size // 2,
                        dtype=torch.uint8,
                    ),
                    requires_grad=False,
                ),
            )

            self.register_buffer(
                "quantized_gate_up_proj_values",
                nn.Parameter(
                    torch.zeros(
                        self.num_experts,
                        self.intermediate_size * 2,
                        self.hidden_size // 2,
                        dtype=torch.uint8,
                    ),
                    requires_grad=False,
                ),
            )

            self.register_buffer(
                "quantized_down_proj_scale_factors",
                nn.Parameter(
                    torch.zeros(
                        self.num_experts,
                        self.hidden_size
                        * self.intermediate_size
                        // self.config.dtype.block_size(),
                        dtype=self.config.dtype.scale_dtype(),
                    ),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "quantized_gate_up_proj_scale_factors",
                nn.Parameter(
                    torch.zeros(
                        self.num_experts,
                        self.hidden_size
                        * (self.intermediate_size * 2)
                        // self.config.dtype.block_size(),
                        dtype=self.config.dtype.scale_dtype(),
                    ),
                    requires_grad=False,
                ),
            )

            self.register_buffer(
                "quantized_down_proj_amax",
                nn.Parameter(
                    torch.zeros(self.num_experts, 1, dtype=torch.float32),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "quantized_gate_up_proj_amax",
                nn.Parameter(
                    torch.zeros(self.num_experts, 1, dtype=torch.float32),
                    requires_grad=False,
                ),
            )

            self.register_buffer(
                "quantized_down_proj_metadata",
                nn.Parameter(
                    torch.zeros(self.num_experts, 4, dtype=torch.int32),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "quantized_gate_up_proj_metadata",
                nn.Parameter(
                    torch.zeros(self.num_experts, 4, dtype=torch.int32),
                    requires_grad=False,
                ),
            )

        self.alpha = 1.702
        self.limit = 7.0

    @property
    def parameters_to_quantize(self) -> tuple[str, ...]:
        """Return high precision parameters to be quantized and deleted."""
        return ("down_proj", "gate_up_proj")

    def get_packing_factor(self, parameter_name: str) -> float:
        """Get the packing factor for a parameter."""
        return (
            2
            if parameter_name
            in {"quantized_down_proj_values", "quantized_gate_up_proj_values"}
            else 1
        )

    def get_quantized_parameters(
        self,
        parameter_name: str,
        parameter: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Prepare this layer for post-training quantization by quantizing the weight,
        storing the quantized weight, and deleting the original weight. This should not
        be done if the layer is used for training, as training requires storage of the
        high-precision weight.
        """

        weight_config = QuantizationConfig(
            backend=self.config.quantize_backend,
            dtype=self.config.dtype,
            scale_rule=self.config.weight_scale_rule,
        )

        quantized_proj = []
        for e in range(parameter.shape[0]):
            q = quantize_to_fp4(parameter[e], weight_config)
            quantized_proj.append(q)

        if "down" in parameter_name:
            prefix = "down"
        elif "gate_up" in parameter_name:
            prefix = "gate_up"

        return {
            f"quantized_{prefix}_proj_values": torch.stack(
                [tensor.values for tensor in quantized_proj],
                dim=0,
            ),
            f"quantized_{prefix}_proj_scale_factors": torch.stack(
                [tensor.scale_factors for tensor in quantized_proj],
                dim=0,
            ),
            f"quantized_{prefix}_proj_amax": torch.stack(
                [tensor.amax for tensor in quantized_proj],
                dim=0,
            ),
            f"quantized_{prefix}_proj_metadata": torch.stack(
                [
                    torch.tensor(
                        [
                            tensor.original_shape[0],
                            tensor.original_shape[1],
                            tensor.padded_shape[0],
                            tensor.padded_shape[1],
                        ],
                    )
                    for tensor in quantized_proj
                ],
            ),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_indices: torch.Tensor = None,
        routing_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for the FP4 experts layer."""

        down_proj, gate_up_proj = self.quantized_weights()

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            -1,
            self.hidden_size,
        )
        next_states = torch.zeros_like(
            hidden_states,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                routing_indices,
                num_classes=self.num_experts,
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for [expert_idx] in expert_hit:
            if expert_idx == self.num_experts:
                continue
            with torch.no_grad():
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # Gate-up projection
            fprop_activation_config = QuantizationConfig(
                backend=self.config.quantize_backend,
                dtype=self.config.dtype,
                scale_rule=self.config.activation_scale_rule,
            )

            gate_up = fp4_matmul(
                current_state,
                gate_up_proj[expert_idx],
                input_config=fprop_activation_config,
                out_dtype=self.config.output_dtype,
            )
            gate_up += self.gate_up_proj_bias[expert_idx]

            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            gated_output = (up + 1) * glu

            # Down projection
            out = fp4_matmul(
                gated_output,
                down_proj[expert_idx],
                input_config=fprop_activation_config,
                out_dtype=self.config.output_dtype,
            )
            out += self.down_proj_bias[expert_idx]
            weighted_output = out * routing_weights[token_idx, top_k_pos, None]
            next_states.index_add_(
                0,
                token_idx,
                weighted_output.to(hidden_states.dtype),
            )

        return next_states.view(batch_size, -1, self.hidden_size)

    def quantized_weights(self) -> tuple[list[QuantizedTensor], list[QuantizedTensor]]:
        """Return quantized parameters as QuantizedTensor."""

        if not hasattr(self, "_quantized_weights"):
            weight_config = self.config.get_weight_config()
            if self.config.keep_master_weights:
                down = [
                    quantize_to_fp4(self.down_proj[e], weight_config)
                    for e in range(self.num_experts)
                ]
                gate_up = [
                    quantize_to_fp4(self.gate_up_proj[e], weight_config)
                    for e in range(self.num_experts)
                ]
                return (down, gate_up)

            down = []
            gate_up = []
            for e in range(self.num_experts):
                down.append(
                    QuantizedTensor(
                        values=self.quantized_down_proj_values.data[e],
                        scale_factors=self.quantized_down_proj_scale_factors.data[e],
                        amax=self.quantized_down_proj_amax.data[e],
                        dtype=self.config.dtype,
                        original_shape=tuple(
                            self.quantized_down_proj_metadata.data[e, :2].tolist(),
                        ),
                        scale_rule=self.config.weight_scale_rule,
                        padded_shape=tuple(
                            self.quantized_down_proj_metadata.data[e, 2:].tolist(),
                        ),
                    ),
                )
                gate_up.append(
                    QuantizedTensor(
                        values=self.quantized_gate_up_proj_values.data[e],
                        scale_factors=self.quantized_gate_up_proj_scale_factors.data[e],
                        amax=self.quantized_gate_up_proj_amax.data[e],
                        dtype=self.config.dtype,
                        original_shape=tuple(
                            self.quantized_gate_up_proj_metadata.data[e, :2].tolist(),
                        ),
                        scale_rule=self.config.weight_scale_rule,
                        padded_shape=tuple(
                            self.quantized_gate_up_proj_metadata.data[e, 2:].tolist(),
                        ),
                    ),
                )
            self._quantized_weights = (down, gate_up)

        return self._quantized_weights
