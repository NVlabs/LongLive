// Long-WAM file attribution; existing upstream notices are retained below.
// SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
// SPDX-FileCopyrightText: Copyright (c) 2023, Tri Dao.
// SPDX-FileCopyrightText: Copyright (c) 2024, Tri Dao.
// SPDX-License-Identifier: MIT AND BSD-3-Clause
// Provenance: Copied from the source snapshot listed below.
// Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/fouroversix/src/fouroversix/csrc/include/hardware_info.h
// Source: https://github.com/mit-han-lab/fouroversix :: src/fouroversix/csrc/include/hardware_info.h
// Source: https://github.com/Dao-AILab/flash-attention
// Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.
// License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
// End Long-WAM attribution.

/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

#if !defined(__CUDACC_RTC__)
#include "cuda_runtime.h"
#endif

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = call;                                                \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(status_));                                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)


inline int get_current_device() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    return device;
}

inline std::tuple<int, int> get_compute_capability(int device) {
    int capability_major, capability_minor;
    CHECK_CUDA(cudaDeviceGetAttribute(&capability_major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&capability_minor, cudaDevAttrComputeCapabilityMinor, device));
    return {capability_major, capability_minor};
}

inline int get_num_sm(int device) {
    int multiprocessor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
    return multiprocessor_count;
}
