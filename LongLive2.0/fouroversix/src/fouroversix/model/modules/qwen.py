import warnings
from typing import Any

import torch
from fouroversix.matmul import fp4_matmul
from fouroversix.model.config import ModuleQuantizationConfig
from fouroversix.model.quantize import QuantizedModule
from fouroversix.quantize import QuantizationConfig, QuantizedTensor, quantize_to_fp4
from torch import nn

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts
except ImportError:
    warnings.warn(
        "Qwen3_5MoeExperts not found, please update transformers to the latest "
        "version to quantize Qwen3.5 models.",
        stacklevel=2,
    )

    Qwen3_5MoeExperts = None


@QuantizedModule.register(Qwen3_5MoeExperts)
class FourOverSixQwenExperts(nn.Module):
    """
    Drop-in replacement for the Qwen3_5MoeExperts layer that
    uses FP4 quantization.
    """

    def __init__(
        self,
        module: Qwen3_5MoeExperts,
        config: ModuleQuantizationConfig,
    ) -> None:
        """
        Initialize the FourOverSixQwenExperts layer.

        Args:
            module (GptOssMLP): The high-precision module that this quantized layer will
                replace.
            config (ModuleQuantizationConfig): The quantization configuration to use for
                the layer.

        """
        super().__init__()

        self.num_experts = module.num_experts
        self.intermediate_dim = module.intermediate_dim
        self.hidden_dim = module.hidden_dim

        self.down_proj = module.down_proj
        self.gate_up_proj = module.gate_up_proj

        self.device = self.down_proj.device
        self.config = config

        self.act_fn = module.act_fn

        if not self.config.keep_master_weights:

            self.register_buffer(
                "quantized_down_proj_values",
                nn.Parameter(
                    torch.zeros(
                        self.num_experts,
                        self.hidden_dim,
                        self.intermediate_dim // 2,
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
                        self.intermediate_dim * 2,
                        self.hidden_dim // 2,
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
                        self.hidden_dim
                        * self.intermediate_dim
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
                        self.hidden_dim
                        * (self.intermediate_dim * 2)
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

    @property
    def parameters_to_quantize(self) -> tuple[str, ...]:
        """Return high precision parameters to be quantized and deleted."""
        return ("down_proj", "gate_up_proj")

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

        if "bias" in parameter_name:
            return {parameter_name: parameter}

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
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the FP4 experts layer."""

        down_proj, gate_up_proj = self.quantized_weights()

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index,
                num_classes=self.num_experts,
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for [expert_idx] in expert_hit:
            if expert_idx == self.num_experts:
                continue

            fprop_activation_config = QuantizationConfig(
                backend=self.config.quantize_backend,
                dtype=self.config.dtype,
                scale_rule=self.config.activation_scale_rule,
            )
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = fp4_matmul(
                current_state,
                gate_up_proj[expert_idx],
                input_config=fprop_activation_config,
                out_dtype=self.config.output_dtype,
            ).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = fp4_matmul(
                current_hidden_states,
                down_proj[expert_idx],
                input_config=fprop_activation_config,
                out_dtype=self.config.output_dtype,
            )
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )

        return final_hidden_states

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
