/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_launch_template.h
 * Copyright (c) 2025, FourOverSix Team.
 ******************************************************************************/

#pragma once
#include <c10/cuda/CUDAException.h> // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "hardware_info.h"
#include "fp4_quant.h"
#include "fp4_quant_kernel.h"

namespace fouroversix
{

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FourOverSix requires building with sm version sm100, but was built for < 10.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FP4_QUANT_KERNEL(kernelName, ...)   \
    template <typename Kernel_traits, __VA_ARGS__> \
    __global__ void kernelName(KERNEL_PARAM_MODIFIER const FP4_quant_params params)

    DEFINE_FP4_QUANT_KERNEL(fp4_quant_prologue_kernel, bool Is_nvfp4, bool Is_rht, bool Is_2d, bool Is_transpose, bool Is_rtn, int kSelectionRule)
    {
#if defined(ARCH_SUPPORTS_FLASH)
        fouroversix::compute_fp4_quant_prologue<Kernel_traits, Is_nvfp4, Is_rht, Is_2d, Is_transpose, Is_rtn, kSelectionRule>(params);
#else
        FLASH_UNSUPPORTED_ARCH
#endif
    }

    DEFINE_FP4_QUANT_KERNEL(fp4_quant_kernel, bool Is_nvfp4, bool Is_rht, bool Is_2d, bool Is_transpose, bool Is_rtn, int kSelectionRule)
    {
#if defined(ARCH_SUPPORTS_FLASH)
        fouroversix::compute_fp4_quant<Kernel_traits, Is_nvfp4, Is_rht, Is_2d, Is_transpose, Is_rtn, kSelectionRule>(params);
#else
        FLASH_UNSUPPORTED_ARCH
#endif
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Kernel_traits, bool Is_nvfp4, bool Is_rht, bool Is_transpose>
    void launch_fp4_quant_prologue(FP4_quant_params &params, cudaStream_t stream)
    {
        constexpr size_t smem_size = Kernel_traits::kSmemSize;

        const int num_m_block = (params.M + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
        const int num_n_block = (params.N + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
        dim3 grid(num_m_block, num_n_block);
        BOOL_SWITCH(params.is_rtn, Is_rtn, [&]
        {
            BOOL_SWITCH(params.is_2d, Is_2d, [&] 
            {
                SELECTION_RULE_SWITCH(params.selection_rule, kSelectionRule, [&]
                {
                    auto kernel_prologue = &fp4_quant_prologue_kernel<Kernel_traits, Is_nvfp4, Is_rht, Is_2d, Is_transpose, Is_rtn, kSelectionRule>;
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel_prologue, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    kernel_prologue<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Kernel_traits, bool Is_nvfp4, bool Is_rht, bool Is_transpose>
    void launch_fp4_quant(FP4_quant_params &params, cudaStream_t stream)
    {
        constexpr size_t smem_size = Kernel_traits::kSmemSize;

        const int num_m_block = (params.M + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
        const int num_n_block = (params.N + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
        dim3 grid(num_m_block, num_n_block);
        BOOL_SWITCH(params.is_rtn, Is_rtn, [&]
        {
            BOOL_SWITCH(params.is_2d, Is_2d, [&] 
            {
                SELECTION_RULE_SWITCH(params.selection_rule, kSelectionRule, [&]
                {
                    auto kernel = &fp4_quant_kernel<Kernel_traits, Is_nvfp4, Is_rht, Is_2d, Is_transpose, Is_rtn, kSelectionRule>;
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // template<int kBlockM_, int kBlockN_, int kNWarps_, bool Is_nvfp4, bool Is_transpose, typename elem_type=cutlass::half_t, typename Base=Base_kernel_traits<elem_type>>

    template <typename T, bool Is_transpose>
    void run_mxfp4_quant(FP4_quant_params &params, cudaStream_t stream)
    {
        constexpr bool Is_nvfp4 = false;
        constexpr bool Is_rht = false;
        launch_fp4_quant_prologue<FP4_quant_kernel_traits<128, 128, 4, Is_nvfp4, Is_transpose, T>, Is_nvfp4, Is_rht, Is_transpose>(params, stream);
        launch_fp4_quant<FP4_quant_kernel_traits<128, 128, 4, Is_nvfp4, Is_transpose, T>, Is_nvfp4, Is_rht, Is_transpose>(params, stream);
    }

    template <typename T, bool Is_transpose>
    void run_mxfp4_quant_rht(FP4_quant_params &params, cudaStream_t stream)
    {
        constexpr bool Is_nvfp4 = false;
        constexpr bool Is_rht = true;
        launch_fp4_quant_prologue<FP4_quant_kernel_traits<128, 128, 4, Is_nvfp4, Is_transpose, T>, Is_nvfp4, Is_rht, Is_transpose>(params, stream);
        launch_fp4_quant<FP4_quant_kernel_traits<128, 128, 4, Is_nvfp4, Is_transpose, T>, Is_nvfp4, Is_rht, Is_transpose>(params, stream);
    }

    template <typename T, bool Is_transpose>
    void run_nvfp4_quant(FP4_quant_params &params, cudaStream_t stream)
    {
        constexpr bool Is_nvfp4 = true;
        constexpr bool Is_rht = false;
        launch_fp4_quant_prologue<FP4_quant_kernel_traits<128, 64, 4, Is_nvfp4, Is_transpose, T>, Is_nvfp4, Is_rht, Is_transpose>(params, stream);
        launch_fp4_quant<FP4_quant_kernel_traits<128, 64, 4, Is_nvfp4, Is_transpose, T>, Is_nvfp4, Is_rht, Is_transpose>(params, stream);
    }

    template <typename T, bool Is_transpose>
    void run_nvfp4_quant_rht(FP4_quant_params &params, cudaStream_t stream)
    {
        constexpr bool Is_nvfp4 = true;
        constexpr bool Is_rht = true;
        launch_fp4_quant_prologue<FP4_quant_kernel_traits<128, 64, 4, Is_nvfp4, Is_transpose, T>, Is_nvfp4, Is_rht, Is_transpose>(params, stream);
        launch_fp4_quant<FP4_quant_kernel_traits<128, 64, 4, Is_nvfp4, Is_transpose, T>, Is_nvfp4, Is_rht, Is_transpose>(params, stream);
    }

} // namespace fouroversix
