// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
// Licensed under the Apache License, Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
//
// No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
//
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

TORCH_LIBRARY(longlive_kernels, m)
{
    m.def("dequantize_kv_cache_fp4(Tensor[] values, Tensor[] scale_factors, Tensor[] amax, int num_heads, int block_token_size, int dtype_code, float e2m1_max, float e4m3_max) -> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "LongLive custom CUDA kernels";
}
