from typing import Any
import torch
import torch.nn as nn
from fouroversix.matmul import fp4_matmul
from fouroversix.model.config import ModuleQuantizationConfig
from fouroversix.model.quantize import QuantizedModule
from fouroversix.quantize import (
                                    QuantizationConfig,
                                    QuantizedTensor,
                                    quantize_to_fp4,
                                )


class FourOverSixLinearFunction(torch.autograd.Function):

    """Differentiable FP4 linear layer."""
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        config: ModuleQuantizationConfig,
        input: torch.Tensor,
        weight: torch.Tensor | QuantizedTensor,
        weight_t: QuantizedTensor | None,
        bias: torch.Tensor = None,
    ) -> tuple[torch.Tensor,]:
        """
        Perform an FP4 matrix multiplication. The input is provided in high precision
        and quantized to FP4 prior to the matrix multiplication, while the weight is
        provided in low precision.
        """
        needs_wgrad = isinstance(weight, (nn.Parameter, torch.Tensor)) and weight.requires_grad
        needs_input_grad = input.requires_grad
        ctx.config = config
        ctx.needs_wgrad = needs_wgrad
        ctx.needs_input_grad_flag = needs_input_grad
        ctx.saved_quantized_weight = None
        ctx.saved_quantized_weight_t = None
        ctx.saved_bias = None
        fprop_activation_config = config.get_activation_config()
        fprop_weight_config = config.get_weight_config()
        if isinstance(weight, QuantizedTensor):
            weight_q = weight
        else:
            weight_q = quantize_to_fp4(weight.data if isinstance(weight, nn.Parameter) else weight, fprop_weight_config)
        input_2d = input.reshape(-1, input.shape[-1])
        input_q = quantize_to_fp4(input_2d, fprop_activation_config)
        if needs_wgrad:
            ctx.save_for_backward(
                input_q.values, input_q.scale_factors, input_q.amax,
                weight, bias,
            )
            ctx.input_shape = input.shape
            ctx.input_q_meta = (
                input_q.original_shape, input_q.padded_shape,
                input_q.dtype, input_q.scale_rule,
            )
        elif needs_input_grad:
            # When master weights are dropped, `weight` is a QuantizedTensor rather than
            # a Tensor/Parameter, so it cannot be passed to save_for_backward.
            if isinstance(weight, QuantizedTensor):
                ctx.saved_quantized_weight = weight
                ctx.saved_quantized_weight_t = weight_t
                ctx.saved_bias = bias
            else:
                ctx.save_for_backward(weight, bias)
            ctx.input_shape = input.shape
        out = fp4_matmul(
            input_q,
            weight_q,
            backend=config.matmul_backend,
            out_dtype=config.output_dtype,
        ).reshape(*input.shape[:-1], weight_q.original_shape[0])
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Backward pass for the FP4 linear layer."""
        if not ctx.needs_wgrad and not ctx.needs_input_grad_flag:
            return None, None, None, None, None

        if ctx.needs_wgrad:
            iq_vals, iq_sf, iq_amax, weight, bias = ctx.saved_tensors
            weight_t = None
        else:
            if ctx.saved_quantized_weight is not None:
                weight = ctx.saved_quantized_weight
                weight_t = ctx.saved_quantized_weight_t
                bias = ctx.saved_bias
            else:
                weight, bias = ctx.saved_tensors
                weight_t = None

        input_shape = ctx.input_shape

        grad_input = None
        if ctx.needs_input_grad_flag:
            dgrad_grad_config = ctx.config.get_gradient_config()
            dgrad_weight_config = ctx.config.get_weight_config(transpose=True)
            dgrad_weight = weight_t if weight_t is not None else weight
            if isinstance(weight, QuantizedTensor) and weight_t is None:
                raise RuntimeError(
                    "Materialized quantized linear layers need cached transposed weights for grad_input. "
                    "Re-materialize with cache_transposed_weights=True."
                )
            grad_input = fp4_matmul(
                grad_output.reshape(-1, grad_output.shape[-1]),
                dgrad_weight,
                backend=ctx.config.matmul_backend,
                input_config=dgrad_grad_config,
                other_config=dgrad_weight_config,
                out_dtype=ctx.config.output_dtype,
            ).reshape(input_shape)

        grad_weight = None
        if ctx.needs_wgrad:
            orig_shape, padded_shape, q_dtype, q_scale_rule = ctx.input_q_meta
            input_approx = QuantizedTensor(
                iq_vals, iq_sf, iq_amax,
                q_dtype, orig_shape, q_scale_rule, padded_shape,
            ).dequantize_triton(dtype=torch.bfloat16)

            wgrad_grad_config = ctx.config.get_gradient_config(rht=True, transpose=True)
            wgrad_activation_config = ctx.config.get_activation_config(
                rht=True,
                transpose=True,
            )
            grad_weight = fp4_matmul(
                grad_output.reshape(-1, grad_output.shape[-1]),
                input_approx,
                backend=ctx.config.matmul_backend,
                input_config=wgrad_grad_config,
                other_config=wgrad_activation_config,
                out_dtype=ctx.config.output_dtype,
            ).unsqueeze(0)

        grad_bias = (
            grad_output.sum(0) if bias is not None and ctx.needs_input_grad[4] else None
        )

        return (
            None,
            grad_input,
            grad_weight,
            None,
            grad_bias,
        )

@QuantizedModule.register(nn.Linear)
class FourOverSixLinear(nn.Linear):
    """
    Drop-in replacement for `nn.Linear` that quantizes weights, activations, and
    gradients.
    """
    def __init__(
        self,
        module: nn.Linear,
        config: ModuleQuantizationConfig,
    ) -> None:
        """
        Initialize the FourOverSixLinear layer.
        Args:
            module (nn.Linear): The high-precision module that this quantized layer will
                replace.
            config (ModuleQuantizationConfig): The quantization configuration to use for
                the layer.
        """
        super().__init__(
            module.in_features,
            module.out_features,
            module.bias is not None,
            module.weight.device,
            module.weight.dtype,
        )
        self.weight = module.weight
        self.bias = module.bias
        self.config = config
        if not self.config.keep_master_weights:
            self.register_buffer(
                "quantized_weight_values",
                nn.Parameter(
                    torch.zeros(
                        self.out_features,
                        self.in_features // 2,
                        dtype=torch.uint8,
                    ),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "quantized_weight_scale_factors",
                nn.Parameter(
                    torch.zeros(
                        self.out_features
                        * self.in_features
                        // self.config.dtype.block_size(),
                        dtype=self.config.dtype.scale_dtype(),
                    ),
                    requires_grad=False,
                ),
            )
            self.register_buffer(
                "quantized_weight_amax",
                nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False),
            )
            self.register_buffer(
                "quantized_weight_metadata",
                nn.Parameter(
                    torch.zeros(2 + 2, dtype=torch.int32),
                    requires_grad=False,
                ),
            )

    @property
    def parameters_to_quantize(self) -> tuple[str, ...]:
        """Return high precision parameters to be quantized and deleted."""
        return ("weight",)

    def get_element_size(self, parameter_name: str) -> float:
        """Get the size of a single element, in bytes, for a parameter."""
        # quantized_weight_values is packed, so there are 4 bits, or 0.5 bytes, per
        # element. Once quantized, weight will have (8+1)/16 bytes per element (one
        # block of 16 values is 8 bytes of values + 1 byte of scale factors).
        return {"quantized_weight_values": 0.5, "weight": 9 / 16}.get(
            parameter_name,
            getattr(self, parameter_name).element_size(),
        )
    
    def get_quantized_parameters(
        self,
        parameter_name: str,
        parameter: torch.Tensor,
        include_transposed: bool = False,
    ) -> dict[str, Any]:
        """Get the quantized parameters for the layer."""
        if parameter_name == "weight":
            config = QuantizationConfig(
                backend=self.config.quantize_backend,
                block_scale_2d=self.config.weight_scale_2d,
                dtype=self.config.dtype,
                scale_rule=self.config.weight_scale_rule,
            )
            quantized_weight = quantize_to_fp4(parameter, config)
            quantized_params = self._serialize_quantized_weight(
                "quantized_weight",
                quantized_weight,
            )
            if include_transposed:
                transposed_config = QuantizationConfig(
                    backend=self.config.quantize_backend,
                    block_scale_2d=self.config.weight_scale_2d,
                    dtype=self.config.dtype,
                    scale_rule=self.config.weight_scale_rule,
                    transpose=True,
                )
                quantized_params.update(
                    self._serialize_quantized_weight(
                        "quantized_weight_transposed",
                        quantize_to_fp4(parameter, transposed_config),
                    ),
                )
            return quantized_params
        msg = f"Unsupported high-preciison parameter: {parameter_name}"
        raise ValueError(msg)

    @staticmethod
    def _serialize_quantized_weight(
        prefix: str,
        quantized_weight: QuantizedTensor,
    ) -> dict[str, torch.Tensor]:
        return {
            f"{prefix}_values": quantized_weight.values,
            f"{prefix}_scale_factors": quantized_weight.scale_factors,
            f"{prefix}_amax": quantized_weight.amax,
            f"{prefix}_metadata": torch.tensor(
                [
                    quantized_weight.original_shape[0],
                    quantized_weight.original_shape[1],
                    quantized_weight.padded_shape[0],
                    quantized_weight.padded_shape[1],
                ],
                dtype=torch.int32,
            ),
        }

    def _build_quantized_weight(self, prefix: str) -> QuantizedTensor | None:
        values = getattr(self, f"{prefix}_values", None)
        scale_factors = getattr(self, f"{prefix}_scale_factors", None)
        amax = getattr(self, f"{prefix}_amax", None)
        metadata = getattr(self, f"{prefix}_metadata", None)
        if any(x is None for x in (values, scale_factors, amax, metadata)):
            return None
        original_shape = tuple(metadata.data[:2].tolist())
        padded_shape = tuple(metadata.data[2:].tolist())
        return QuantizedTensor(
            values.data,
            scale_factors.data,
            amax.data,
            self.config.dtype,
            original_shape,
            self.config.weight_scale_rule,
            padded_shape,
        )

    def quantized_weight(self) -> QuantizedTensor:
        """
        Prepare this layer for post-training quantization by quantizing the weight,
        storing the quantized weight, and deleting the original weight. This should not
        be done if the layer is used for training, as training requires storage of the
        high-precision weight.
        """
        if not hasattr(self, "_quantized_weight"):
            if self.config.keep_master_weights:
                return self.weight
            self._quantized_weight = self._build_quantized_weight("quantized_weight")
        return self._quantized_weight

    def quantized_weight_transposed(self) -> QuantizedTensor | None:
        if self.config.keep_master_weights:
            return None
        if not hasattr(self, "_quantized_weight_transposed"):
            self._quantized_weight_transposed = self._build_quantized_weight(
                "quantized_weight_transposed",
            )
        return self._quantized_weight_transposed
        
    _INT32_MAX = 2_147_483_647

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FP4 linear layer."""
        flat_rows = input.reshape(-1, input.shape[-1]).shape[0]
        if flat_rows * input.shape[-1] > self._INT32_MAX:
            return self._forward_chunked(input)
        return FourOverSixLinearFunction.apply(
            self.config,
            input,
            self.quantized_weight(),
            self.quantized_weight_transposed(),
            self.bias,
        )

    def _forward_chunked(self, input: torch.Tensor) -> torch.Tensor:
        """Split evenly into 2 chunks to avoid int32 overflow in CUDA FP4 quantize kernel."""
        orig_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1])
        mid = input_2d.shape[0] // 2
        weight = self.quantized_weight()
        weight_t = self.quantized_weight_transposed()
        out_a = FourOverSixLinearFunction.apply(self.config, input_2d[:mid], weight, weight_t, self.bias)
        out_b = FourOverSixLinearFunction.apply(self.config, input_2d[mid:], weight, weight_t, self.bias)
        return torch.cat([out_a, out_b], dim=0).reshape(*orig_shape[:-1], -1)