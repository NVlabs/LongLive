// Long-WAM file attribution; existing upstream notices are retained below.
// SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
// SPDX-License-Identifier: MIT
// Provenance: Copied from the source snapshot listed below.
// Source: https://github.com/mit-han-lab/fouroversix @ 4c0db4bf3027272c61e32ca7e2cdb9c7fa5bcecb :: src/fouroversix/csrc/quantize/fp4_quant_fp16_mxfp4_rht_sm100.cu
// Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.
// License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
// Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: third_party/fouroversix/src/fouroversix/csrc/quantize/fp4_quant_fp16_mxfp4_rht_sm100.cu
// End Long-WAM attribution.

// Splitting the different transpose modes to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "fp4_quant_launch_template.h"
namespace fouroversix {

template<>
void run_fp4_quant_<cutlass::half_t, false, true, false>(FP4_quant_params &params, cudaStream_t stream) {
    run_mxfp4_quant_rht<cutlass::half_t, false>(params, stream);
}

} // namespace fouroversix
