// Long-WAM file attribution; existing upstream notices are retained below.
// SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// SPDX-FileCopyrightText: Copyright (c) 2025, FourOverSix Team.
// SPDX-License-Identifier: MIT AND Apache-2.0
// Provenance: Copied from the source snapshot listed below.
// Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/fouroversix/src/fouroversix/csrc/include/static_switch.h
// Source: https://github.com/mit-han-lab/fouroversix :: src/fouroversix/csrc/include/static_switch.h
// Source: https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.
// License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
// End Long-WAM attribution.

// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

// Adapted by Junxian Guo from https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// Copyright (c) 2025, FourOverSix Team.

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    } }()

#define FP16_SWITCH(COND, ...) \
  [&] {                                      \
    if (COND) {                              \
      using fp16_type = cutlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using fp16_type = cutlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    } }()

#define SELECTION_RULE_SWITCH(SELECTION_RULE, ...) \
  [&] {                                              \
    if (SELECTION_RULE == 0) {                       \
      constexpr static int kSelectionRule = 0;       \
      return __VA_ARGS__();                          \
    } else if (SELECTION_RULE == 1) {                \
      constexpr static int kSelectionRule = 1;       \
      return __VA_ARGS__();                          \
    } else if (SELECTION_RULE == 2) {                \
      constexpr static int kSelectionRule = 2;       \
      return __VA_ARGS__();                          \
    } else if (SELECTION_RULE == 3) {                \
      constexpr static int kSelectionRule = 3;       \
      return __VA_ARGS__();                          \
    } else if (SELECTION_RULE == 4) {                \
      constexpr static int kSelectionRule = 4;       \
      return __VA_ARGS__();                          \
    } else {                                         \
      constexpr static int kSelectionRule = 0;       \
      return __VA_ARGS__();                          \
    } }()
