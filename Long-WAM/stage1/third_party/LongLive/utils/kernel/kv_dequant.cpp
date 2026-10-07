// Long-WAM file attribution; existing upstream notices are retained below.
// SPDX-FileCopyrightText: The LongLive contributors
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: Apache-2.0
// Provenance: Copied from the source snapshot listed below.
// Source: https://github.com/NVlabs/LongLive @ 0308b126accba9440b8caa45bcf7bec0877933e1 :: utils/kernel/kv_dequant.cpp
// Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/utils/kernel/kv_dequant.cpp
// License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
// End Long-WAM attribution.

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
