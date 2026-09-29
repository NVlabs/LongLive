from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

SRC_DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}


SM = [100]  # Sm100 kernels support up to
IS_NVFP4 = ["false", "true"]
IS_TRANSPOSE = ["false", "true"]
IS_RHT = ["false", "true"]


def get_fp4_quant_template(
    is_nvfp4: str,
    is_rht: str,
    is_transpose: str,
    src_dtype: str,
) -> str:
    if is_nvfp4 == "false":
        function_str = "run_mxfp4_quant"
    elif is_nvfp4 == "true":
        function_str = "run_nvfp4_quant"
    else:
        msg = f"Invalid is_nvfp4: {is_nvfp4}"
        raise ValueError(msg)

    if is_rht == "true":
        function_str = f"{function_str}_rht"

    return f"""#include "fp4_quant_launch_template.h"
namespace fouroversix {{

template<>
void run_fp4_quant_<{src_dtype}, {is_nvfp4}, {is_rht}, {is_transpose}>(FP4_quant_params &params, cudaStream_t stream) {{
    {function_str}<{src_dtype}, {is_transpose}>(params, stream);
}}

}} // namespace fouroversix"""  # noqa: E501


@dataclass
class Kernel:
    """Representation for a kernel that quantizes a tensor to FP4."""

    sm: int
    src_dtype: str
    is_nvfp4: str
    is_rht: str
    is_transpose: str
    op: str

    @property
    def template(self) -> str:
        """The kernel's template content."""

        template_funcs = {
            "fp4_quant": get_fp4_quant_template,
        }
        template_func = template_funcs[self.op]
        return template_func(
            is_transpose=self.is_transpose,
            src_dtype=SRC_DTYPE_MAP[self.src_dtype],
            is_nvfp4=self.is_nvfp4,
            is_rht=self.is_rht,
        )

    @property
    def filename(self) -> str:
        """The filename for the kernel."""

        fp4_format = "nvfp4" if self.is_nvfp4 == "true" else "mxfp4"
        return (
            f"{self.op}_{self.src_dtype}_{fp4_format}_"
            f"{'rht_' if self.is_rht == 'true' else ''}"
            f"{'trans_' if self.is_transpose == 'true' else ''}sm{self.sm}.cu"
        )


def get_all_kernels() -> list[Kernel]:
    for op in ["fp4_quant"]:
        for src_dtype, is_nvfp4, is_rht, is_transpose, sm in itertools.product(
            SRC_DTYPE_MAP.keys(),
            IS_NVFP4,
            IS_RHT,
            IS_TRANSPOSE,
            SM,
        ):
            yield Kernel(
                sm=sm,
                src_dtype=src_dtype,
                is_rht=is_rht,
                is_nvfp4=is_nvfp4,
                is_transpose=is_transpose,
                op=op,
            )


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Splitting the different transpose modes to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n"""  # noqa: E501
    content = prelude + kernel.template
    (autogen_dir / kernel.filename).write_text(content)


def main(output_dir: str | None) -> None:
    if output_dir is None:
        output_dir = (
            Path(__file__).parent.parent / "src" / "fouroversix" / "csrc" / "quantize"
        )
    else:
        output_dir = Path(output_dir)

    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the kernels  will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)
