/******************************************************************************
 * Copyright (c) 2025, FourOverSix Team.
 ******************************************************************************/

#pragma once
#include <torch/extension.h>

namespace fouroversix
{

    enum AdaptiveBlockScalingRuleType
    {
        STATIC_6 = 0,
        STATIC_4 = 1,
        MAE_4o6 = 2,
        MSE_4o6 = 3,
        ABS_MAX_4o6 = 4,
    };

    struct FP4_quant_params
    {
        using index_t = int64_t;
        void *__restrict__ x_ptr;
        void *__restrict__ x_rht_ptr;
        void *__restrict__ x_e2m1_ptr;
        void *__restrict__ x_sf_ptr;
        void *__restrict__ x_sft_ptr;
        void *__restrict__ amax_ptr;

        int x_row_stride;
        int x_col_stride;
        int x_rht_row_stride;
        int x_rht_col_stride;
        int x_e2m1_row_stride;
        int x_e2m1_col_stride;
        int x_sf_row_stride;
        int x_sf_col_stride;
        int x_sft_row_stride;
        int x_sft_col_stride;

        // The dimensions.
        int M, N, M_rounded, N_rounded, M_sf, N_sf;
        bool is_bf16;
        bool is_nvfp4;
        bool is_rtn;
        bool is_rht;
        bool is_4o6;
        bool is_2d;
        bool is_transpose;
        int selection_rule; // 0: static_6, 1: static_4, 2: 4o6_mae, 3: 4o6_mse
        int rbits;
    };

    template <typename T, bool Is_nvfp4, bool Is_rht, bool Is_transpose>
    void run_fp4_quant_(FP4_quant_params &params, cudaStream_t stream);

} // namespace fouroversix
