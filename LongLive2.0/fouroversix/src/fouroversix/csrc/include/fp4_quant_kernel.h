/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h
 * Copyright (c) 2025, FourOverSix Team.
 ******************************************************************************/

#pragma once

// #include "philox_unpack.cuh" // For at::cuda::philox::unpack

#include <cute/tensor.hpp>
#include <type_traits>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "kernel_traits.h"
#include "utils.h"
#include "hadamard_transform.h"
// #include "softmax.h"
// #include "mask.h"
// #include "dropout.h"
// #include "rotary.h"

namespace fouroversix
{

    using namespace cute;

    template <typename Kernel_traits, bool Is_nvfp4, bool Is_rht, bool Is_2d, bool Is_transpose, bool Is_rtn, int kSelectionRule, typename Params>
    inline __device__ void compute_fp4_quant_prologue_block(const Params &params, const int m_block, const int n_block)
    {
        // Type aliases
        using Element = typename Kernel_traits::Element;
        using ScaleFactor = typename Kernel_traits::ScaleFactor;
        using index_t = typename Kernel_traits::index_t;

        // Compile-time constants
        constexpr int kGroupN = Kernel_traits::kGroupN;
        constexpr int kBlockM = Kernel_traits::kBlockM;
        constexpr int kBlockN = Kernel_traits::kBlockN;
        constexpr int kNWarps = Kernel_traits::kNWarps;
        constexpr int kNumGroupsInRow = Kernel_traits::kNumGroupsInRow;
        constexpr int kNumGroupsInCol = Kernel_traits::kNumGroupsInCol;
        constexpr float E4M3_MAX_VALUE = Kernel_traits::E4M3_MAX_VALUE;
        constexpr float E2M1_MAX_VALUE = Kernel_traits::E2M1_MAX_VALUE;

        constexpr AdaptiveBlockScalingRuleType kRule = static_cast<AdaptiveBlockScalingRuleType>(kSelectionRule);
        constexpr bool Is_4o6 = kRule == AdaptiveBlockScalingRuleType::MAE_4o6 ||
                                kRule == AdaptiveBlockScalingRuleType::MSE_4o6 ||
                                kRule == AdaptiveBlockScalingRuleType::ABS_MAX_4o6;

        using VecTypeX = cutlass::Array<Element, kGroupN>;
        using VecTypeXFloat = cutlass::Array<float, kGroupN>;
        using VecTypeSFT = cutlass::Array<float, 4>;
        constexpr int kVecSizeSFT = 4;

        // Shared memory
        extern __shared__ char smem[];

        // Runtime variables
        const int tidx = threadIdx.x;
        const int num_groups = kNumGroupsInRow * kBlockM;
        float *amax_ptr = reinterpret_cast<float *>(params.amax_ptr);

        // -------------------------------------------------------------------------
        // Tensor Definitions
        // -------------------------------------------------------------------------

        // Input X (Global Memory)
        Tensor mX = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.x_ptr)),
                                make_shape(params.M, params.N),
                                make_stride(params.x_row_stride, _1{}));
        Tensor gX = local_tile(mX(_, _), Shape<Int<kBlockM>, Int<kBlockN>>{},
                               make_coord(m_block, n_block));

        Tensor mXRHT = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.x_rht_ptr)),
                                   make_shape(params.M_rounded, params.N_rounded),
                                   make_stride(params.x_rht_row_stride, _1{}));
        Tensor gXRHT = local_tile(mXRHT(_, _), Shape<Int<kBlockM>, Int<kBlockN>>{},
                                  make_coord(m_block, n_block));

        // Scale Factor Temp SFT (Global Memory)
        Tensor mSFT = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(params.x_sft_ptr)),
                                  make_shape(params.M, params.N_rounded / kGroupN),
                                  make_stride(params.x_sft_row_stride, _1{}));
        Tensor gSFT = local_tile(mSFT(_, _), Shape<Int<kBlockM>, Int<kBlockN / kGroupN>>{},
                                 make_coord(m_block, n_block));

        // Shared Memory Tensors
        Tensor sX = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem)),
                                typename Kernel_traits::SmemLayoutX{});

        // SFT in Shared Memory (placed after X)
        Tensor sSFT = make_tensor(make_smem_ptr(reinterpret_cast<float *>(reinterpret_cast<char *>(sX.data().get()) + sizeof(Element) * size(sX))),
                                  typename Kernel_traits::SmemLayoutSFT{});

        // -------------------------------------------------------------------------
        // Data Loading (X -> Shared)
        // -------------------------------------------------------------------------

        typename Kernel_traits::GmemTiledCopyX gmem_tiled_copy_X;
        auto gmem_thr_copy_X = gmem_tiled_copy_X.get_thread_slice(tidx);

        Tensor tXgX = gmem_thr_copy_X.partition_S(gX);
        Tensor tXsX = gmem_thr_copy_X.partition_D(sX);

        // Construct predicates for bounds checking
        Tensor cX = make_identity_tensor(make_shape(size<0>(sX), size<1>(sX)));
        Tensor tXcX = gmem_thr_copy_X.partition_S(cX);
        Tensor tXpX = make_tensor<bool>(make_shape(size<2>(tXcX)));

        for (int i = 0; i < size(tXpX); ++i)
        {
            tXpX(i) = get<1>(tXcX(0, 0, i)) < params.N - n_block * kBlockN;
        }

        __syncthreads();

        // Async copy from Global to Shared
        fouroversix::copy<false, false, true /*Clear_OOB_MN*/, true /*Clear_OOB_K*/>(
            gmem_tiled_copy_X, tXgX, tXsX, tXcX, tXpX, params.M - m_block * kBlockM);

        cute::cp_async_fence();
        fouroversix::cp_async_wait<0>();
        __syncthreads();

        // -------------------------------------------------------------------------
        // Scale Factor Computation
        // -------------------------------------------------------------------------

        float thr_max = 0.0f;
        for (int g_idx = tidx; g_idx < num_groups; g_idx += blockDim.x)
        {
            const int g_row = g_idx / kNumGroupsInRow;
            const int g_col = g_idx % kNumGroupsInRow;

            VecTypeXFloat x_vec_float;
            for (int i = 0; i < kGroupN; ++i)
            {
                x_vec_float[i] = static_cast<float>(sX(g_row, g_col * kGroupN + i));
            }
            if constexpr (Is_rht)
            {
                hadamard_quant_group<Is_nvfp4, Element>(&x_vec_float[0]);
                VecTypeX x_vec;
#pragma unroll
                for (int i = 0; i < kGroupN; ++i)
                {
                    // sX(g_row, g_col * kGroupN + i) = static_cast<Element>(x_vec_float[i]);
                    x_vec[i] = static_cast<Element>(x_vec_float[i]);
                }

                *reinterpret_cast<VecTypeX *>(&gXRHT(g_row, g_col * kGroupN)) = *reinterpret_cast<VecTypeX *>(&x_vec);
            }
            // VecTypeX x_vec = *reinterpret_cast<VecTypeX *>(&sX(g_row, g_col * kGroupN));

            // Compute max absolute value in group
            float sf = 0.0f;
#pragma unroll
            for (int i = 0; i < kGroupN; ++i)
            {
                sf = max(sf, abs(x_vec_float[i]));
            }

            thr_max = max(thr_max, sf);
            sSFT(g_row, g_col) = sf;
        }

        if constexpr (Is_2d)
        {
            __syncthreads();
        }

        if constexpr (Is_2d)
        {
            MaxOp<float> max_op;
            for (int g_idx = tidx; g_idx < num_groups; g_idx += blockDim.x)
            {
                const int g_row = g_idx % kNumGroupsInCol;
                const int g_col = g_idx / kNumGroupsInCol;
                float sf = sSFT(g_row, g_col);
                float blk_sf = Allreduce<kGroupN>::run(sf, max_op); // kGroupN is 16 or 32
                sSFT(g_row, g_col) = blk_sf;
                __syncthreads();
            }
        }

        // -------------------------------------------------------------------------
        // Normalization Constant Reduction (Block-wide Max)
        // -------------------------------------------------------------------------

        // Warp-level reduction
        MaxOp<float> max_op;
        float warp_max = Allreduce<32>::run(thr_max, max_op);

        // Block-level reduction via shared memory
        float *sRed = reinterpret_cast<float *>(smem);
        if (tidx % 32 == 0)
        {
            sRed[tidx / 32] = warp_max;
        }
        __syncthreads();

        if (tidx == 0)
        {
            float blk_max = 0.0f;
#pragma unroll
            for (int i = 0; i < kNWarps; ++i)
            {
                blk_max = max(blk_max, sRed[i]);
            }
            atomicMaxFloat(amax_ptr, blk_max);
        }

        // -------------------------------------------------------------------------
        // Write Back SFT (Shared -> Global)
        // -------------------------------------------------------------------------

        for (int r_idx = tidx; r_idx < kBlockM; r_idx += blockDim.x)
        {
#pragma unroll
            for (int i = 0; i < int(kBlockN / kGroupN); i += kVecSizeSFT)
            {
                *reinterpret_cast<VecTypeSFT *>(&gSFT(r_idx, i)) = *reinterpret_cast<VecTypeSFT *>(&sSFT(r_idx, i));
            }
        }
    }

    template <typename Kernel_traits, bool Is_nvfp4, bool Is_rht, bool Is_2d, bool Is_transpose, bool Is_rtn, int kSelectionRule, typename Params>
    inline __device__ void compute_fp4_quant_prologue(const Params &params)
    {
        // TODO: Implement the fp4 quant kernel
        const int m_block = blockIdx.x;
        // The block index for the batch.
        const int n_block = blockIdx.y;

        fouroversix::compute_fp4_quant_prologue_block<Kernel_traits, Is_nvfp4, Is_rht, Is_2d, Is_transpose, Is_rtn, kSelectionRule>(params, m_block, n_block);
    }

    template <typename Kernel_traits, bool Is_nvfp4, bool Is_rht, bool Is_2d, bool Is_transpose, bool Is_rtn, int kSelectionRule, typename Params>
    inline __device__ void compute_fp4_quant_block(const Params &params, const int m_block, const int n_block)
    {
        // Type aliases
        using Element = typename Kernel_traits::Element;
        using ScaleFactor = typename Kernel_traits::ScaleFactor;
        using index_t = typename Kernel_traits::index_t;

        // Compile-time constants
        constexpr int kGroupN = Kernel_traits::kGroupN;
        constexpr int kBlockM = Kernel_traits::kBlockM;
        constexpr int kBlockN = Kernel_traits::kBlockN;
        constexpr int kBlockMSF = Kernel_traits::kBlockMSF;
        constexpr int kBlockNSF = Kernel_traits::kBlockNSF;
        constexpr int kNWarps = Kernel_traits::kNWarps;
        constexpr int kNumGroupsInRow = Kernel_traits::kNumGroupsInRow;
        constexpr int kNumGroupsInCol = Kernel_traits::kNumGroupsInCol;
        constexpr float E4M3_MAX_VALUE = Kernel_traits::E4M3_MAX_VALUE;

        constexpr AdaptiveBlockScalingRuleType kRule = static_cast<AdaptiveBlockScalingRuleType>(kSelectionRule);
        constexpr bool Is_4o6 = kRule == AdaptiveBlockScalingRuleType::MAE_4o6 ||
                                kRule == AdaptiveBlockScalingRuleType::MSE_4o6 ||
                                kRule == AdaptiveBlockScalingRuleType::ABS_MAX_4o6;
        constexpr float E4M3_SCALE_4 = Is_4o6 ? Kernel_traits::E4M3_MAX_FOUROVERSIX : E4M3_MAX_VALUE;
        constexpr float E4M3_SCALE_6 = Is_4o6 ? Kernel_traits::E4M3_MAX_FOUROVERSIX : E4M3_MAX_VALUE;
        constexpr float E2M1_SCALE_4 = Is_4o6 ? 6.0f : 4.0f;
        constexpr float E2M1_SCALE_6 = 6.0f;

        constexpr int kSmemBlockInRow = int(kNumGroupsInRow / 4);
        constexpr int kSmemBlockInCol = int(kBlockM / 128);

        using VecTypeXe2m1 = std::conditional_t<Is_nvfp4, cutlass::Array<uint8_t, 8>, cutlass::Array<uint8_t, 16>>;
        using VecTypeSFT = cutlass::Array<float, 4>;
        using VecTypeSF = cutlass::Array<ScaleFactor, 16>;
        using OutputType = cutlass::Array<cutlass::float_e2m1_t, 8>;
        constexpr int kVecSizeXe2m1 = Is_nvfp4 ? 8 : 16;
        constexpr int kVecSizeSFT = 4;
        constexpr int kVecSizeSF = 16;

        // Shared memory
        extern __shared__ char smem[];

        // Runtime variables
        const int tidx = threadIdx.x;
        const int num_groups = kNumGroupsInRow * kBlockM;
        // JXGuo: assure amax is not zero before calling this kernel
        const float amax = *reinterpret_cast<float *>(params.amax_ptr);

        if (amax == 0.0f)
        {
            return;
        }

        // -------------------------------------------------------------------------
        // Tensor Definitions
        // -------------------------------------------------------------------------

        // Input X (Global Memory)
        // void *__restrict__ x_ptr = Is_rht ? params.x_rht_ptr : params.x_ptr;
        Tensor mX = Is_rht ? make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.x_rht_ptr)),
                                         make_shape(params.M_rounded, params.N_rounded),
                                         make_stride(params.x_rht_row_stride, _1{}))
                           : make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.x_ptr)),
                                         make_shape(params.M, params.N),
                                         make_stride(params.x_row_stride, _1{}));
        Tensor gX = local_tile(mX(_, _), Shape<Int<kBlockM>, Int<kBlockN>>{},
                               make_coord(m_block, n_block));

        Tensor mXe2m1 = make_tensor(make_gmem_ptr(reinterpret_cast<uint8_t *>(params.x_e2m1_ptr)),
                                    make_shape(params.M, params.N_rounded / 2),
                                    make_stride(params.x_e2m1_row_stride, _1{}));
        Tensor gXe2m1 = local_tile(mXe2m1(_, _), Shape<Int<kBlockM>, Int<kBlockN / 2>>{},
                                   make_coord(m_block, n_block));

        // Scale Factor Temp SFT (Global Memory)
        Tensor mSFT = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(params.x_sft_ptr)),
                                  make_shape(params.M, params.N_rounded / kGroupN),
                                  make_stride(params.x_sft_row_stride, _1{}));
        Tensor gSFT = local_tile(mSFT(_, _), Shape<Int<kBlockM>, Int<kBlockN / kGroupN>>{},
                                 make_coord(m_block, n_block));

        Tensor gSF = make_tensor(make_gmem_ptr(reinterpret_cast<ScaleFactor *>(params.x_sf_ptr)),
                                 make_shape(params.M_sf, params.N_sf),
                                 make_stride(params.x_sf_row_stride, _1{}));
        // Tensor gSF = local_tile(mSF(_, _), Shape<Int<1>, Int<16>>{},
        //                          make_coord(m_block, n_block));

        // Shared Memory Tensors
        Tensor sX = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem)),
                                typename Kernel_traits::SmemLayoutX{});

        // SFT in Shared Memory (placed after X)
        Tensor sSFT = make_tensor(make_smem_ptr(reinterpret_cast<float *>(reinterpret_cast<char *>(sX.data().get()) + sizeof(Element) * size(sX))),
                                  typename Kernel_traits::SmemLayoutSFT{});

        Tensor sXe2m1 = make_tensor(make_smem_ptr(reinterpret_cast<uint8_t *>(reinterpret_cast<char *>(sSFT.data().get()) + sizeof(float) * size(sSFT))),
                                    Shape<Int<kBlockM>, Int<kBlockN / 2>>{},
                                    Stride<Int<kBlockN / 2>, _1>{});

        Tensor sSF = make_tensor(make_smem_ptr(reinterpret_cast<ScaleFactor *>(reinterpret_cast<char *>(sXe2m1.data().get()) + sizeof(uint8_t) * size(sXe2m1))),
                                 typename Kernel_traits::SmemLayoutSF{});

        // -------------------------------------------------------------------------
        // Data Loading (X -> Shared)
        // -------------------------------------------------------------------------

        typename Kernel_traits::GmemTiledCopyX gmem_tiled_copy_X;
        auto gmem_thr_copy_X = gmem_tiled_copy_X.get_thread_slice(tidx);

        Tensor tXgX = gmem_thr_copy_X.partition_S(gX);
        Tensor tXsX = gmem_thr_copy_X.partition_D(sX);

        // Construct predicates for bounds checking
        Tensor cX = make_identity_tensor(make_shape(size<0>(sX), size<1>(sX)));
        Tensor tXcX = gmem_thr_copy_X.partition_S(cX);
        Tensor tXpX = make_tensor<bool>(make_shape(size<2>(tXcX)));

        for (int i = 0; i < size(tXpX); ++i)
        {
            tXpX(i) = get<1>(tXcX(0, 0, i)) < params.N - n_block * kBlockN;
        }

        __syncthreads();

        // Async copy from Global to Shared
        fouroversix::copy<false, false, true /*Clear_OOB_MN*/, true /*Clear_OOB_K*/>(
            gmem_tiled_copy_X, tXgX, tXsX, tXcX, tXpX, params.M - m_block * kBlockM);

        cute::cp_async_fence();

        // -------------------------------------------------------------------------
        // Data Loading (SFT -> Shared)
        // -------------------------------------------------------------------------

        for (int r_idx = tidx; r_idx < kBlockM; r_idx += blockDim.x)
        {
#pragma unroll
            for (int i = 0; i < int(kBlockN / kGroupN); i += kVecSizeSFT)
            {
                *reinterpret_cast<VecTypeSFT *>(&sSFT(r_idx, i)) = *reinterpret_cast<VecTypeSFT *>(&gSFT(r_idx, i));
            }
        }

        fouroversix::cp_async_wait<0>();
        __syncthreads();

        // -------------------------------------------------------------------------
        // Quantization
        // -------------------------------------------------------------------------

        for (int g_idx = tidx; g_idx < num_groups; g_idx += blockDim.x)
        {
            const int g_row = g_idx % kNumGroupsInCol;
            const int g_col = g_idx / kNumGroupsInCol;
            const float g_max = sSFT(g_row, g_col);

            const Tensor sGX = make_tensor(make_smem_ptr(sX.data() + g_row * kBlockN + g_col * kGroupN),
                                           Shape<Int<1>, Int<kGroupN>>{},
                                           Stride<Int<kGroupN>, _1>{});

            OutputType res[int(kGroupN / 8)];
            float encode_scale;
            float sf;

            if constexpr (Is_4o6)
            {
                encode_scale = E2M1_SCALE_6 * E4M3_SCALE_6 / amax;

                float sf_high_precision = g_max / E2M1_SCALE_6 * encode_scale;
                float sf_[2] = {sf_high_precision * 1.5, sf_high_precision};

                sf_[0] = static_cast<float>(static_cast<ScaleFactor>(sf_[0]));
                sf_[1] = static_cast<float>(static_cast<ScaleFactor>(sf_[1]));

                sf = fp4_conversion<Is_nvfp4, Is_2d, true, Is_rtn, kRule>(sGX, amax, sf_, res, params.rbits);
            }
            else
            {
                float sf_val = 0.0f;
                if constexpr (kRule == AdaptiveBlockScalingRuleType::STATIC_6)
                {
                    encode_scale = E4M3_SCALE_6 * E2M1_SCALE_6 / amax;
                    sf_val = clamp(g_max / E2M1_SCALE_6 * encode_scale, 0, E4M3_MAX_VALUE);
                }
                else if constexpr (kRule == AdaptiveBlockScalingRuleType::STATIC_4)
                {
                    encode_scale = E2M1_SCALE_4 * E4M3_SCALE_4 / amax;
                    sf_val = clamp(g_max / E2M1_SCALE_4 * encode_scale, 0, E4M3_MAX_VALUE);
                }
                else
                {
                    printf("in fp4_quant_block, kRule = %d, not supported\n", kRule);
                    assert(false);
                }

                sf_val = static_cast<float>(static_cast<ScaleFactor>(sf_val));
                sf = fp4_conversion<Is_nvfp4, false, false, Is_rtn, kRule>(sGX, amax, &sf_val, res, params.rbits);
            }

            // Write quantized data
            for (int i = 0; i < int(kGroupN / 8); ++i)
            {
                *reinterpret_cast<OutputType *>(&sXe2m1(g_row, g_col * (kGroupN / 2) + i * 4)) = res[i];
            }

            // Write scale factor (layout: 128x4 blocks, 32 rows per block)
            const int r_in_blk = g_row % 128;
            const int c_in_blk = g_col % 4;
            const int blk_row = int(g_row / 128);
            const int blk_col = int(g_col / 4);
            const int sf_row = 32 * (blk_row * kSmemBlockInRow + blk_col) + r_in_blk % 32;
            const int sf_col = int(r_in_blk / 32) * 4 + c_in_blk;
            sSF(sf_row, sf_col) = static_cast<ScaleFactor>(sf);
            __syncthreads();
        }

        // -------------------------------------------------------------------------
        // Write Back Xe2m1 (Shared -> Global)
        // -------------------------------------------------------------------------

        __syncthreads();

        for (int r_idx = tidx; r_idx < kBlockM; r_idx += blockDim.x)
        {
#pragma unroll
            for (int i = 0; i < int(kBlockN / 2); i += kVecSizeXe2m1)
            {
                *reinterpret_cast<VecTypeXe2m1 *>(&gXe2m1(r_idx, i)) = *reinterpret_cast<VecTypeXe2m1 *>(&sXe2m1(r_idx, i));
            }
        }

        // -------------------------------------------------------------------------
        // Write Back SF (Shared -> Global)
        // -------------------------------------------------------------------------

        const int gbl_blk_row_stride = int(params.N_rounded / (kGroupN * 4));
        const int gbl_blk_col_stride = 1;
        const int gbl_blk_idx_base = (m_block * kSmemBlockInCol) * gbl_blk_row_stride + (n_block * kSmemBlockInRow) * gbl_blk_col_stride;

        static_assert(kVecSizeSF == kBlockNSF, "kVecSizeSF must be equal to kBlockNSF");
        for (int r_idx = tidx; r_idx < kBlockMSF; r_idx += blockDim.x)
        {
            const int loc_blk_idx = int(r_idx / 32);
            const int loc_row = r_idx % 32;
            const int loc_blk_row = int(loc_blk_idx / kSmemBlockInRow);
            const int loc_blk_col = int(loc_blk_idx % kSmemBlockInRow);
            const int gbl_blk_idx = gbl_blk_idx_base + loc_blk_row * gbl_blk_row_stride + loc_blk_col * gbl_blk_col_stride;
            const index_t gbl_row = index_t(32) * gbl_blk_idx + loc_row;
            *reinterpret_cast<VecTypeSF *>(&gSF(gbl_row, 0)) = *reinterpret_cast<VecTypeSF *>(&sSF(r_idx, 0));
        }
    }

    template <typename Kernel_traits, bool Is_nvfp4, bool Is_rht, bool Is_2d, bool Is_transpose, bool Is_rtn, int kSelectionRule, typename Params>
    inline __device__ void compute_fp4_quant(const Params &params)
    {
        // TODO: Implement the fp4 quant kernel
        const int m_block = blockIdx.x;
        // The block index for the batch.
        const int n_block = blockIdx.y;

        fouroversix::compute_fp4_quant_block<Kernel_traits, Is_nvfp4, Is_rht, Is_2d, Is_transpose, Is_rtn, kSelectionRule>(params, m_block, n_block);
    }

} // namespace fouroversix
