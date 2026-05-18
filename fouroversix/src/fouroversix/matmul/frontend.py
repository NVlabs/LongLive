import torch
from fouroversix.quantize import QuantizationConfig, QuantizedTensor, quantize_to_fp4
from fouroversix.utils import DataType, MatmulBackend

from .cutlass import CUTLASSMatmulBackend
from .pytorch import PyTorchMatmulBackend

AVAILABLE_BACKENDS = {
    MatmulBackend.cutlass: CUTLASSMatmulBackend,
    MatmulBackend.pytorch: PyTorchMatmulBackend,
}


def fp4_matmul(
    input: torch.Tensor | QuantizedTensor,
    other: torch.Tensor | QuantizedTensor,
    *,
    backend: MatmulBackend | None = None,
    input_config: QuantizationConfig | None = None,
    other_config: QuantizationConfig | None = None,
    out_dtype: DataType = DataType.bfloat16,
) -> torch.Tensor:
    """
    Perform a matrix multiplication (`a @ b.T`) between two quantized tensors.

    ## Sample Code

    Each tensor may be provided in either high or low precision. If provided in high
    precision, tensors will be quantized to FP4 prior to the matrix multiplication, and
    quantization may be configured with the `input_quantize_kwargs` and
    `other_quantize_kwargs` parameters. For example, the following two code samples are
    equivalent:

    ### With High-Precision Inputs

    ```python
    a = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    b = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    out = fp4_matmul(a, b)
    ```

    ### With Low-Precision Inputs

    ```python
    a = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")
    b = torch.tensor(1024, 1024, dtype=torch.bfloat16, device="cuda")

    a_quantized = quantize_to_fp4(a)
    b_quantized = quantize_to_fp4(b)

    out = fp4_matmul(a_quantized, b_quantized)
    ```

    ## Backends

    We provide two different implementations of FP4 matrix multiplication:

    - **CUTLASS**: Uses CUTLASS kernels to perform fast FP4 matrix multiplication.
        Requires a Blackwell GPU.
    - **PyTorch**: A slow implementation which dequantizes FP4 tensors and then
        performs a high-precision matrix multiplication.

    ## Parameters

    Args:
        input (torch.Tensor | QuantizedTensor): The first tensor to be multiplied.
        other (torch.Tensor | QuantizedTensor): The second tensor to be multiplied.
        backend (MatmulBackend): The backend to use for the matrix multiplication,
            either `MatmulBackend.cutlass` or `MatmulBackend.pytorch`. If no backend is
            provided, CUTLASS will be used if the machine has a Blackwell GPU, and
            PyTorch will be used otherwise.
        input_config (QuantizationConfig | None): If `input` is provided in high
            precision, this configuration will be passed to the `quantize_to_fp4` call
            done prior to the matrix multiplication.
        other_config (QuantizationConfig | None): If `other` is provided in high
            precision, this configuration will be passed to the `quantize_to_fp4` call
            done prior to the matrix multiplication.
        out_dtype (DataType): The data type of the output tensor. Defaults to
            `DataType.bfloat16`.

    Returns:
        The output tensor.

    """

    if input_config is None:
        input_config = QuantizationConfig()

    if isinstance(input, torch.Tensor):
        input = quantize_to_fp4(input, input_config)

    if other_config is None:
        other_config = QuantizationConfig()

    if isinstance(other, torch.Tensor):
        other = quantize_to_fp4(other, other_config)

    if backend is None:
        for backend_candidate in [MatmulBackend.cutlass, MatmulBackend.pytorch]:
            if AVAILABLE_BACKENDS[backend_candidate] is not None and AVAILABLE_BACKENDS[
                backend_candidate
            ].is_supported(input, other, out_dtype=out_dtype):
                backend = backend_candidate
                break
        else:
            msg = "No backend found that supports the given parameters"
            raise ValueError(msg)

    elif not AVAILABLE_BACKENDS[backend].is_supported(
        input, other, out_dtype=out_dtype,
    ):
        msg = f"Backend {backend} does not support the given parameters"
        raise ValueError(msg)

    return AVAILABLE_BACKENDS[backend].fp4_matmul(input, other, out_dtype=out_dtype)
