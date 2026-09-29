import functools

import torch
from fouroversix.quantize.utils import to_blocked
from fouroversix.utils import BLACKWELL_SM_IDS, DataType, RoundStyle, ScaleRule

from .backend import QuantizeBackendBase
from .config import QuantizationConfig
from .quantized_tensor import QuantizedTensor


class TransformerEngineQuantizeBackend(QuantizeBackendBase):
    """
    A backend that quantizes inputs using NVIDIA's TransformerEngine. Used to debug
    our implementations.
    """

    @classmethod
    @functools.lru_cache
    def is_available(cls) -> bool:
        """
        Return True if the Transformer Engine backend is available on the current
        machine.
        """

        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] not in BLACKWELL_SM_IDS
        ):
            return False

        try:
            import transformer_engine  # noqa: F401
        except ModuleNotFoundError:
            return False

        return True

    @classmethod
    def is_supported(cls, x: torch.Tensor, config: QuantizationConfig) -> bool:
        """
        Return True if the Transformer Engine backend supports the given input and
        quantization configuration.
        """

        if not super().is_supported(x, config):
            return False

        if config.dtype != DataType.nvfp4 or config.scale_rule != ScaleRule.static_6:
            return False

        if not config.transpose and config.rht:
            return False

        if config.transpose and config.rht and config.block_scale_2d:  # noqa: SIM103
            return False

        return True

    @classmethod
    def quantize_to_fp4(
        cls,
        x: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedTensor:
        """
        Quantize a tensor to FP4 using the Transformer Engine backend.

        Args:
            x (torch.Tensor): The input tensor to quantize.
            config (QuantizationConfig): The quantization configuration.

        Returns:
            The quantized tensor.

        """

        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

        q = NVFP4Quantizer(
            with_2d_quantization=config.block_scale_2d,
            with_rht=config.rht,
            with_post_rht_amax=config.rht,
            stochastic_rounding=config.round_style == RoundStyle.stochastic,
        )

        out = q.quantize(x)

        if config.transpose:
            values = out._columnwise_data  # noqa: SLF001
            scale_factors = to_blocked(
                out._columnwise_scale_inv.view(torch.float8_e4m3fn),  # noqa: SLF001
            )
            amax = out._amax_columnwise  # noqa: SLF001
        else:
            values = out._rowwise_data  # noqa: SLF001
            scale_factors = to_blocked(
                out._rowwise_scale_inv.view(torch.float8_e4m3fn),  # noqa: SLF001
            )
            amax = out._amax_rowwise  # noqa: SLF001

        return QuantizedTensor(
            values,
            scale_factors,
            amax,
            config.dtype,
            (x.shape[1], x.shape[0]) if config.transpose else x.shape,
            config.scale_rule,
        )
