import torch
from fouroversix.utils import QuantizeBackend

from .config import QuantizationConfig
from .cuda import CUDAQuantizeBackend
from .pytorch import PyTorchQuantizeBackend
from .quantized_tensor import QuantizedTensor
from .transformer_engine import TransformerEngineQuantizeBackend
from .triton import TritonQuantizeBackend

AVAILABLE_BACKENDS = {
    QuantizeBackend.cuda: CUDAQuantizeBackend,
    QuantizeBackend.transformer_engine: TransformerEngineQuantizeBackend,
    QuantizeBackend.triton: TritonQuantizeBackend,
    QuantizeBackend.pytorch: PyTorchQuantizeBackend,
}


def quantize_to_fp4(
    x: torch.Tensor,
    config: QuantizationConfig | None = None,
) -> QuantizedTensor:
    """
    Quantize a tensor to FP4.

    ## Sample Code

    ### With Four Over Six

    ```python
    x = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    x_quantized = quantize_to_fp4(x)
    ```

    ### Without Four Over Six

    ```python
    x = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(scale_rule="static_6")
    x_quantized = quantize_to_fp4(x, config)
    ```

    ### With Stochastic Rounding

    ```python
    x = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(round_style="stochastic")
    x_quantized = quantize_to_fp4(x, config)
    ```

    ### With the Random Hadamard Transform

    ```python
    from fouroversix.quantize import get_rht_matrix

    x = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    config = QuantizationConfig(rht=True)
    x_quantized = quantize_to_fp4(x, config)
    ```

    ## Backends

    We provide three different implementations of FP4 quantization:

    - **CUDA**: A fast implementation written in CUDA which currently only supports
        basic quantization options (no 2D block scaling, no stochastic rounding, no
        random Hadamard transform). Can be used for inference, but not training.
        Requires a Blackwell GPU.
    - **Triton**: A slightly slower implementation written in Triton which supports all
        operations needed for training. Also requires a Blackwell GPU.
    - **PyTorch**: A slow implementation written in PyTorch which supports all
        operations and can be run on any GPU.

    If `quantize_to_fp4` is called with `backend=None`, a backend will be selected
    automatically based on the following rules:

    - If there is no GPU available, or if the available GPU is not a Blackwell GPU,
        select PyTorch.
    - If any quantization options are set other than `scale_rule`, select Triton.
        - However, if the available GPU is SM120 (i.e. RTX 5090, RTX 6000) and
            `round_style` is set to `RoundStyle.stochastic`, select PyTorch as
            stochastic rounding does not have hardware support on SM120 GPUs.
    - Otherwise, select CUDA.

    ## Parameters

    Args:
        x (torch.Tensor): The input tensor to quantize.
        config (QuantizationConfig): The quantization configuration to use. If no
            configuration is provided, a default configuration will be used (NVFP4,
            1D block scaling, round-to-nearest, and 4/6 with the MSE selection rule).

    Returns:
        The quantized tensor.

    """

    if config is None:
        config = QuantizationConfig()

    selected_backend = config.backend

    if selected_backend is None:
        for backend in [
            QuantizeBackend.cuda,
            QuantizeBackend.triton,
            QuantizeBackend.pytorch,
        ]:
            if AVAILABLE_BACKENDS[backend] is not None and AVAILABLE_BACKENDS[
                backend
            ].is_supported(x, config):
                selected_backend = backend
                break
        else:
            msg = "No backend found that supports the given parameters"
            raise ValueError(msg)

    elif not AVAILABLE_BACKENDS[selected_backend].is_supported(x, config):
        msg = f"Backend {selected_backend} does not support the given parameters"
        raise ValueError(msg)

    return AVAILABLE_BACKENDS[selected_backend].quantize_to_fp4(x, config)
