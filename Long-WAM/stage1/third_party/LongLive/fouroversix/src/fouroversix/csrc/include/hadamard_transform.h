// Long-WAM file attribution; existing upstream notices are retained below.
// SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
// SPDX-FileCopyrightText: Copyright (c) 2023, Tri Dao.
// SPDX-FileCopyrightText: Copyright (c) 2025, FourOverSix Team.
// SPDX-License-Identifier: MIT AND BSD-3-Clause
// Provenance: Copied from the source snapshot listed below.
// Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/fouroversix/src/fouroversix/csrc/include/hadamard_transform.h
// Source: https://github.com/mit-han-lab/fouroversix :: src/fouroversix/csrc/include/hadamard_transform.h
// Source: https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/csrc/code_gen.py
// Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.
// License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
// End Long-WAM attribution.


/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/csrc/code_gen.py
 * Copyright (c) 2025, FourOverSix Team.
 ******************************************************************************/

// This file is auto-generated. See "hadamard_code_gen.py"


#pragma once


namespace fouroversix {

template <typename Element>
__device__ __forceinline__ void hadamard_quant_group_16(float x[16]) {
    float out[16];
    out[0] = + x[0] + x[1] + x[2] - x[3] + x[4] - x[5] - x[6] - x[7] - x[8] - x[9] - x[10] + x[11] - x[12] + x[13] - x[14] - x[15];
    out[1] = + x[0] - x[1] + x[2] + x[3] + x[4] + x[5] - x[6] + x[7] - x[8] + x[9] - x[10] - x[11] - x[12] - x[13] - x[14] + x[15];
    out[2] = + x[0] + x[1] - x[2] + x[3] + x[4] - x[5] + x[6] + x[7] - x[8] - x[9] + x[10] - x[11] - x[12] + x[13] + x[14] + x[15];
    out[3] = + x[0] - x[1] - x[2] - x[3] + x[4] + x[5] + x[6] - x[7] - x[8] + x[9] + x[10] + x[11] - x[12] - x[13] + x[14] - x[15];
    out[4] = + x[0] + x[1] + x[2] - x[3] - x[4] + x[5] + x[6] + x[7] - x[8] - x[9] - x[10] + x[11] + x[12] - x[13] + x[14] + x[15];
    out[5] = + x[0] - x[1] + x[2] + x[3] - x[4] - x[5] + x[6] - x[7] - x[8] + x[9] - x[10] - x[11] + x[12] + x[13] + x[14] - x[15];
    out[6] = + x[0] + x[1] - x[2] + x[3] - x[4] + x[5] - x[6] - x[7] - x[8] - x[9] + x[10] - x[11] + x[12] - x[13] - x[14] - x[15];
    out[7] = + x[0] - x[1] - x[2] - x[3] - x[4] - x[5] - x[6] + x[7] - x[8] + x[9] + x[10] + x[11] + x[12] + x[13] - x[14] + x[15];
    out[8] = + x[0] + x[1] + x[2] - x[3] + x[4] - x[5] - x[6] - x[7] + x[8] + x[9] + x[10] - x[11] + x[12] - x[13] + x[14] + x[15];
    out[9] = + x[0] - x[1] + x[2] + x[3] + x[4] + x[5] - x[6] + x[7] + x[8] - x[9] + x[10] + x[11] + x[12] + x[13] + x[14] - x[15];
    out[10] = + x[0] + x[1] - x[2] + x[3] + x[4] - x[5] + x[6] + x[7] + x[8] + x[9] - x[10] + x[11] + x[12] - x[13] - x[14] - x[15];
    out[11] = + x[0] - x[1] - x[2] - x[3] + x[4] + x[5] + x[6] - x[7] + x[8] - x[9] - x[10] - x[11] + x[12] + x[13] - x[14] + x[15];
    out[12] = + x[0] + x[1] + x[2] - x[3] - x[4] + x[5] + x[6] + x[7] + x[8] + x[9] + x[10] - x[11] - x[12] + x[13] - x[14] - x[15];
    out[13] = + x[0] - x[1] + x[2] + x[3] - x[4] - x[5] + x[6] - x[7] + x[8] - x[9] + x[10] + x[11] - x[12] - x[13] - x[14] + x[15];
    out[14] = + x[0] + x[1] - x[2] + x[3] - x[4] + x[5] - x[6] - x[7] + x[8] + x[9] - x[10] + x[11] - x[12] + x[13] + x[14] + x[15];
    out[15] = + x[0] - x[1] - x[2] - x[3] - x[4] - x[5] - x[6] + x[7] + x[8] - x[9] - x[10] - x[11] - x[12] - x[13] + x[14] - x[15];
    #pragma unroll
    for (int i = 0; i < 16; i++) { x[i] = static_cast<float>(static_cast<Element>(out[i] / 4)); }
}

template <typename Element>
__device__ __forceinline__ void hadamard_quant_group_32(float x[32]) {
    hadamard_quant_group_16<Element>(x);
    hadamard_quant_group_16<Element>(x + 16);
}

template <bool Is_nvfp4, typename Element>
__device__ __forceinline__ void hadamard_quant_group(float* x) {
    if constexpr (Is_nvfp4) {
        hadamard_quant_group_16<Element>(x);
    } else {
        hadamard_quant_group_32<Element>(x);
    }
}


} // namespace fouroversix

