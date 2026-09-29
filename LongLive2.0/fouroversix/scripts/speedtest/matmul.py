from __future__ import annotations

from typing import Any

import click
import modal

from ..resources import app, get_image

img = get_image()

with img.imports():
    import torch
    import torch.utils.benchmark as benchmark
    from fouroversix import DataType, MatmulBackend, QuantizationConfig, quantize_to_fp4
    from fouroversix.matmul.frontend import AVAILABLE_BACKENDS


def run_speedtest(
    *,
    dtype: DataType = DataType.nvfp4,
    m: int = 1024,
    n: int = 1024,
    k: int = 1024,
    repeats: int = 100,
) -> None:
    """Test speed on a B200 on Modal."""

    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")

    config = QuantizationConfig(dtype=dtype)
    x_quantized = quantize_to_fp4(x, config)
    y_quantized = quantize_to_fp4(y, config)
    out_dtype = DataType.bfloat16

    print(f"Testing with {m}x{k} @ {k}x{n}")

    for backend in [MatmulBackend.cutlass, MatmulBackend.pytorch]:
        backend_cls = AVAILABLE_BACKENDS[backend]
        print(f"{backend.value}: ", end="")

        if not backend_cls.is_available():
            print("Not available")
            continue

        if not backend_cls.is_supported(
            x_quantized,
            y_quantized,
            out_dtype=out_dtype,
        ):
            print("Not supported")
            continue

        t = benchmark.Timer(
            setup="from fouroversix import fp4_matmul",
            stmt=(
                "fp4_matmul(x_quantized, y_quantized, backend=backend, "
                "out_dtype=out_dtype)"
            ),
            globals={
                "x_quantized": x_quantized,
                "y_quantized": y_quantized,
                "backend": backend,
                "out_dtype": out_dtype,
            },
        )

        print(f"{t.timeit(repeats).mean * 1000:.4f}ms")


@app.function(image=img, cpu=4, memory=8 * 1024, gpu="B200")
def run_speedtest_on_modal(**kwargs: dict[str, Any]) -> None:
    run_speedtest(**kwargs)


@click.command()
@click.option("--dtype", type=DataType, default=DataType.nvfp4)
@click.option("--m", type=int, default=1024)
@click.option("--modal", is_flag=True)
@click.option("--n", type=int, default=1024)
@click.option("--k", type=int, default=1024)
@click.option("--repeats", type=int, default=100)
def cli(**kwargs: dict[str, Any]) -> None:
    if kwargs.pop("modal"):
        with modal.enable_output(), app.run():
            run_speedtest_on_modal.remote(**kwargs)
    else:
        run_speedtest(**kwargs)


if __name__ == "__main__":
    cli()
