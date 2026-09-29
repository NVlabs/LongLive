# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


THIS_DIR = Path(__file__).resolve().parent

setup(
    name="longlive_kv_dequant_cuda",
    ext_modules=[
        CUDAExtension(
            name="longlive_kv_dequant_cuda",
            sources=[
                str(THIS_DIR / "kv_dequant.cpp"),
                str(THIS_DIR / "kv_dequant_cuda.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    # iter-37: need sm_100a (Blackwell arch-specific) for
                    # cvt.rn.f16x2.e2m1x2 instruction. Plain sm_100 lacks it.
                    "-gencode=arch=compute_100a,code=sm_100a",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
