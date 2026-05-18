/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Adapted by Junxian Guo and Jack Cook from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/utils.h
 * Copyright (c) 2025, FourOverSix Team.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>
#include <type_traits>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace fouroversix
{

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    __forceinline__ __device__ uint32_t relu2(const uint32_t x);

    template <>
    __forceinline__ __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x)
    {
        uint32_t res;
        const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
        asm volatile(
            "{\n"
            "\t .reg .f16x2 sela;\n"
            "\t set.gtu.u32.f16x2 sela, %1, %2;\n"
            "\t and.b32 %0, sela, %1;\n"
            "}\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
        return res;
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    template <>
    __forceinline__ __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x)
    {
        uint32_t res;
        const uint32_t zero = 0u;
        asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
        return res;
    }
#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

    template <typename T>
    __forceinline__ __device__ uint32_t convert_relu2(const float2 x);

    template <>
    __forceinline__ __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x)
    {
        uint32_t res;
        const uint32_t a = reinterpret_cast<const uint32_t &>(x.x);
        const uint32_t b = reinterpret_cast<const uint32_t &>(x.y);
        asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
        return res;
    }

    template <>
    __forceinline__ __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x)
    {
        uint32_t res;
        const uint32_t a = reinterpret_cast<const uint32_t &>(x.x);
        const uint32_t b = reinterpret_cast<const uint32_t &>(x.y);
        asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
        return res;
    }

#endif

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct MaxOp
    {
        __device__ __forceinline__ T operator()(T const &x, T const &y) { return x > y ? x : y; }
    };

    template <>
    struct MaxOp<float>
    {
        // This is slightly faster
        __device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    struct SumOp
    {
        __device__ __forceinline__ T operator()(T const &x, T const &y) { return x + y; }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <int THREADS>
    struct Allreduce
    {
        static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
        template <typename T, typename Operator>
        static __device__ __forceinline__ T run(T x, Operator &op)
        {
            constexpr int OFFSET = THREADS / 2;
            x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
            return Allreduce<OFFSET>::run(x, op);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    struct Allreduce<2>
    {
        template <typename T, typename Operator>
        static __device__ __forceinline__ T run(T x, Operator &op)
        {
            x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
            return x;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <bool A_in_regs = false, bool B_in_regs = false, typename Tensor0, typename Tensor1,
              typename Tensor2, typename Tensor3, typename Tensor4,
              typename TiledMma, typename TiledCopyA, typename TiledCopyB,
              typename ThrCopyA, typename ThrCopyB>
    __forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsA,
                                         Tensor4 const &tCsB, TiledMma tiled_mma,
                                         TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                                         ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B)
    {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));  // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));  // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB)); // MMA_K
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view)); // M
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view)); // N
        if (!A_in_regs)
        {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
        }
        if (!B_in_regs)
        {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        }
#pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i)
        {
            if (i < size<2>(tCrA) - 1)
            {
                if (!A_in_regs)
                {
                    cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
                }
                if (!B_in_regs)
                {
                    cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
                }
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
              typename TiledMma, typename TiledCopy, typename ThrCopy>
    __forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const &tCsB,
                                            TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                            ThrCopy smem_thr_copy_B)
    {
        CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));  // MMA_M
        CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));  // MMA_N
        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB)); // MMA_K
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view)); // N
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i)
        {
            if (i < size<2>(tCrA) - 1)
            {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    template <typename Layout>
    __forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout)
    {
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_2>{}); // ((2, 2), MMA_M, MMA_N)
        return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
    template <typename MMA_traits, typename Layout>
    __forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout)
    {
        using X = Underscore;
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
        static_assert(mma_shape_K == 8 || mma_shape_K == 16);
        if constexpr (mma_shape_K == 8)
        {
            return acc_layout;
        }
        else
        {
            auto l = logical_divide(acc_layout, Shape<X, X, _2>{}); // (4, MMA_M, (2, MMA_N / 2)))
            return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    template <typename Layout>
    __forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout)
    {
        using X = Underscore;
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{}); // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename To_type, typename Engine, typename Layout>
    __forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor)
    {
        using From_type = typename Engine::value_type;
        constexpr int numel = decltype(size(tensor))::value;
        cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
        // HACK: this requires tensor to be "contiguous"
        auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
        return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    // Blocks until all but N previous cp.async.commit_group operations have committed.
    // This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
    // (which is equivalent to commit_group then wait_group 0).
    // Instead we just call cp.async.wait_group 0, which is slightly faster.
    // https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
    template <int N>
    CUTE_HOST_DEVICE void cp_async_wait()
    {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false, bool Clear_OOB_K = true,
              typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
              typename Engine2, typename Layout2, typename Engine3, typename Layout3>
    __forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                                         Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                         Tensor<Engine3, Layout3> const &predicate_K, const int max_MN = 0)
    {
        CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
        CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
        CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D)); // MMA
        CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D)); // MMA_M
        CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D)); // MMA_K
        // There's no case where !Clear_OOB_K && Clear_OOB_MN
        static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
#pragma unroll
        for (int m = 0; m < size<1>(S); ++m)
        {
            if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN)
            {
#pragma unroll
                for (int k = 0; k < size<2>(S); ++k)
                {
                    if (Is_even_K || predicate_K(k))
                    {
                        cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                    }
                    else if (Clear_OOB_K)
                    {
                        cute::clear(D(_, m, k));
                    }
                }
            }
            else if (Clear_OOB_MN)
            {
                cute::clear(D(_, m, _));
            }
        }
    }

    static __device__ __forceinline__ float atomicMaxFloat(float *addr, float value)
    {
        // source: https://stackoverflow.com/a/51549250
        return (value >= 0)
                   ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
                   : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <bool Is_4o6, bool Is_rtn, AdaptiveBlockScalingRuleType kAdaptiveBlockScalingRuleType>
    struct Fp4ArrayQuant
    {
        using InputType = cutlass::Array<float, 8>;
        using OutputType = cutlass::Array<cutlass::float_e2m1_t, 8>;
        using ScaleFactorType = float;
        using ErrorType = float;

        __device__ __forceinline__
            OutputType
            convert(InputType const &x,
                    const float amax,
                    const ScaleFactorType sf,
                    const uint32_t rbits,
                    // Usage depends on Is_rtn
                    ErrorType *err /*nullable*/)
        {
            InputType x_scaled;
            constexpr float E2M1_MAX_VALUE = 6.0f;
            constexpr float E2M1_MAX_FOUR = 4.0f;
            constexpr float E4M3_MAX_VALUE = 448.0f;
            constexpr float E4M3_MAX_FOUROVERSIX = 256.0f;

            constexpr float e2m1_limit = kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::STATIC_4 ? E2M1_MAX_FOUR : E2M1_MAX_VALUE;
            constexpr float e4m3_limit = (kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::STATIC_6 || kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::STATIC_4)
                                             ? E4M3_MAX_VALUE
                                             : E4M3_MAX_FOUROVERSIX;
            const float encode_scale = e4m3_limit * e2m1_limit / amax;
            const float decode_scale = 1.0 / encode_scale;
            const float block_scale_inv = fminf(1.0f / (decode_scale * sf), std::numeric_limits<float>::max());

#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                x_scaled[i] = x[i] * block_scale_inv;
            }

            unsigned out;

            if constexpr (Is_rtn)
            {
                if constexpr (Is_4o6)
                {
                    unsigned out_dequant_1;
                    unsigned out_dequant_2;
                    unsigned out_dequant_3;
                    unsigned out_dequant_4;

                    asm volatile(
                        "{\n"
                        ".reg .b8 byte0, byte1, byte2, byte3;\n"
                        "cvt.rn.satfinite.e2m1x2.f32   byte0, %6, %5;\n"
                        "cvt.rn.satfinite.e2m1x2.f32   byte1, %8, %7;\n"
                        "cvt.rn.satfinite.e2m1x2.f32   byte2, %10, %9;\n"
                        "cvt.rn.satfinite.e2m1x2.f32   byte3, %12, %11;\n"
                        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
                        "cvt.rn.f16x2.e2m1x2 %1, byte0;\n"
                        "cvt.rn.f16x2.e2m1x2 %2, byte1;\n"
                        "cvt.rn.f16x2.e2m1x2 %3, byte2;\n"
                        "cvt.rn.f16x2.e2m1x2 %4, byte3;\n"
                        "}"
                        : "=r"(out), "=r"(out_dequant_1), "=r"(out_dequant_2), "=r"(out_dequant_3), "=r"(out_dequant_4) : "f"(x_scaled[0]), "f"(x_scaled[1]), "f"(x_scaled[2]), "f"(x_scaled[3]),
                                                                                                                          "f"(x_scaled[4]), "f"(x_scaled[5]), "f"(x_scaled[6]), "f"(x_scaled[7]));

                    unsigned short out_dequant_1_hi = (out_dequant_1 >> 16) & 0xFFFF;
                    unsigned short out_dequant_1_lo = out_dequant_1 & 0xFFFF;
                    unsigned short out_dequant_2_hi = (out_dequant_2 >> 16) & 0xFFFF;
                    unsigned short out_dequant_2_lo = out_dequant_2 & 0xFFFF;
                    unsigned short out_dequant_3_hi = (out_dequant_3 >> 16) & 0xFFFF;
                    unsigned short out_dequant_3_lo = out_dequant_3 & 0xFFFF;
                    unsigned short out_dequant_4_hi = (out_dequant_4 >> 16) & 0xFFFF;
                    unsigned short out_dequant_4_lo = out_dequant_4 & 0xFFFF;

                    float val0 = __half2float(__ushort_as_half(out_dequant_1_lo)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val1 = __half2float(__ushort_as_half(out_dequant_1_hi)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val2 = __half2float(__ushort_as_half(out_dequant_2_lo)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val3 = __half2float(__ushort_as_half(out_dequant_2_hi)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val4 = __half2float(__ushort_as_half(out_dequant_3_lo)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val5 = __half2float(__ushort_as_half(out_dequant_3_hi)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val6 = __half2float(__ushort_as_half(out_dequant_4_lo)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val7 = __half2float(__ushort_as_half(out_dequant_4_hi)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);

                    if constexpr (kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::MAE_4o6)
                    {
                        *err += std::abs(val0 - x[0]);
                        *err += std::abs(val1 - x[1]);
                        *err += std::abs(val2 - x[2]);
                        *err += std::abs(val3 - x[3]);
                        *err += std::abs(val4 - x[4]);
                        *err += std::abs(val5 - x[5]);
                        *err += std::abs(val6 - x[6]);
                        *err += std::abs(val7 - x[7]);
                    }
                    else if constexpr (kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::MSE_4o6)
                    {
                        *err += (val0 - x[0]) * (val0 - x[0]);
                        *err += (val1 - x[1]) * (val1 - x[1]);
                        *err += (val2 - x[2]) * (val2 - x[2]);
                        *err += (val3 - x[3]) * (val3 - x[3]);
                        *err += (val4 - x[4]) * (val4 - x[4]);
                        *err += (val5 - x[5]) * (val5 - x[5]);
                        *err += (val6 - x[6]) * (val6 - x[6]);
                        *err += (val7 - x[7]) * (val7 - x[7]);
                    }
                    else if constexpr (kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::ABS_MAX_4o6)
                    {
                        float val0_err = std::abs(val0 - x[0]);
                        if (val0_err > *err)
                            *err = val0_err;
                        float val1_err = std::abs(val1 - x[1]);
                        if (val1_err > *err)
                            *err = val1_err;
                        float val2_err = std::abs(val2 - x[2]);
                        if (val2_err > *err)
                            *err = val2_err;
                        float val3_err = std::abs(val3 - x[3]);
                        if (val3_err > *err)
                            *err = val3_err;
                        float val4_err = std::abs(val4 - x[4]);
                        if (val4_err > *err)
                            *err = val4_err;
                        float val5_err = std::abs(val5 - x[5]);
                        if (val5_err > *err)
                            *err = val5_err;
                        float val6_err = std::abs(val6 - x[6]);
                        if (val6_err > *err)
                            *err = val6_err;
                        float val7_err = std::abs(val7 - x[7]);
                        if (val7_err > *err)
                            *err = val7_err;
                    }
                    else
                    {
                        printf("in Fp4ArrayQuant::convert, kAdaptiveBlockScalingRuleType = %d, not supported\n", kAdaptiveBlockScalingRuleType);
                        assert(false);
                    }

                    return reinterpret_cast<OutputType const &>(out);
                }
                else
                {
                    asm volatile(
                        "{\n"
                        ".reg .b8 byte0, byte1, byte2, byte3;\n"
                        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
                        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
                        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
                        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
                        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
                        "}"
                        : "=r"(out) : "f"(x_scaled[0]), "f"(x_scaled[1]), "f"(x_scaled[2]), "f"(x_scaled[3]),
                                      "f"(x_scaled[4]), "f"(x_scaled[5]), "f"(x_scaled[6]), "f"(x_scaled[7]));
                    return reinterpret_cast<OutputType const &>(out);
                }
            }
            else
            {
                if constexpr (Is_4o6)
                {
                    unsigned out_dequant_1;
                    unsigned out_dequant_2;
                    unsigned out_dequant_3;
                    unsigned out_dequant_4;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
                    asm volatile(
                        "{\n"
                        ".reg .b16 tmp0, tmp1;\n"
                        ".reg .b8 byte0, byte1;\n"
                        "cvt.rs.satfinite.e2m1x4.f32   tmp0, {%8, %7, %6, %5}, %13;\n"
                        "mov.b16 {byte1, byte0}, tmp0;\n"
                        "cvt.rn.f16x2.e2m1x2 %1, byte0;\n"
                        "cvt.rn.f16x2.e2m1x2 %2, byte1;\n"
                        "cvt.rs.satfinite.e2m1x4.f32   tmp1, {%12, %11, %10, %9}, %14;\n"
                        "mov.b16 {byte1, byte0}, tmp1;\n"
                        "cvt.rn.f16x2.e2m1x2 %3, byte0;\n"
                        "cvt.rn.f16x2.e2m1x2 %4, byte1;\n"
                        "mov.b32 %0, {tmp0, tmp1};\n"
                        "}"
                        : "=r"(out), "=r"(out_dequant_1), "=r"(out_dequant_2), "=r"(out_dequant_3), "=r"(out_dequant_4) : "f"(x_scaled[0]), "f"(x_scaled[1]), "f"(x_scaled[2]), "f"(x_scaled[3]), "f"(x_scaled[4]), "f"(x_scaled[5]), "f"(x_scaled[6]), "f"(x_scaled[7]), "r"(rbits), "r"(rbits));
#endif

                    unsigned short out_dequant_1_hi = (out_dequant_1 >> 16) & 0xFFFF;
                    unsigned short out_dequant_1_lo = out_dequant_1 & 0xFFFF;
                    unsigned short out_dequant_2_hi = (out_dequant_2 >> 16) & 0xFFFF;
                    unsigned short out_dequant_2_lo = out_dequant_2 & 0xFFFF;
                    unsigned short out_dequant_3_hi = (out_dequant_3 >> 16) & 0xFFFF;
                    unsigned short out_dequant_3_lo = out_dequant_3 & 0xFFFF;
                    unsigned short out_dequant_4_hi = (out_dequant_4 >> 16) & 0xFFFF;
                    unsigned short out_dequant_4_lo = out_dequant_4 & 0xFFFF;

                    float val0 = __half2float(__ushort_as_half(out_dequant_1_lo)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val1 = __half2float(__ushort_as_half(out_dequant_1_hi)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val2 = __half2float(__ushort_as_half(out_dequant_2_lo)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val3 = __half2float(__ushort_as_half(out_dequant_2_hi)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val4 = __half2float(__ushort_as_half(out_dequant_3_lo)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val5 = __half2float(__ushort_as_half(out_dequant_3_hi)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val6 = __half2float(__ushort_as_half(out_dequant_4_lo)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);
                    float val7 = __half2float(__ushort_as_half(out_dequant_4_hi)) * sf * amax / (E2M1_MAX_VALUE * E4M3_MAX_FOUROVERSIX);

                    if constexpr (kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::MAE_4o6)
                    {
                        *err += std::abs(val0 - x[0]);
                        *err += std::abs(val1 - x[1]);
                        *err += std::abs(val2 - x[2]);
                        *err += std::abs(val3 - x[3]);
                        *err += std::abs(val4 - x[4]);
                        *err += std::abs(val5 - x[5]);
                        *err += std::abs(val6 - x[6]);
                        *err += std::abs(val7 - x[7]);
                    }
                    else if constexpr (kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::MSE_4o6)
                    {
                        *err += (val0 - x[0]) * (val0 - x[0]);
                        *err += (val1 - x[1]) * (val1 - x[1]);
                        *err += (val2 - x[2]) * (val2 - x[2]);
                        *err += (val3 - x[3]) * (val3 - x[3]);
                        *err += (val4 - x[4]) * (val4 - x[4]);
                        *err += (val5 - x[5]) * (val5 - x[5]);
                        *err += (val6 - x[6]) * (val6 - x[6]);
                        *err += (val7 - x[7]) * (val7 - x[7]);
                    }
                    else if constexpr (kAdaptiveBlockScalingRuleType == AdaptiveBlockScalingRuleType::ABS_MAX_4o6)
                    {
                        float val0_err = std::abs(val0 - x[0]);
                        if (val0_err > *err)
                            *err = val0_err;
                        float val1_err = std::abs(val1 - x[1]);
                        if (val1_err > *err)
                            *err = val1_err;
                        float val2_err = std::abs(val2 - x[2]);
                        if (val2_err > *err)
                            *err = val2_err;
                        float val3_err = std::abs(val3 - x[3]);
                        if (val3_err > *err)
                            *err = val3_err;
                        float val4_err = std::abs(val4 - x[4]);
                        if (val4_err > *err)
                            *err = val4_err;
                        float val5_err = std::abs(val5 - x[5]);
                        if (val5_err > *err)
                            *err = val5_err;
                        float val6_err = std::abs(val6 - x[6]);
                        if (val6_err > *err)
                            *err = val6_err;
                        float val7_err = std::abs(val7 - x[7]);
                        if (val7_err > *err)
                            *err = val7_err;
                    }
                    else
                    {
                        printf("in Fp4ArrayQuant::convert, kAdaptiveBlockScalingRuleType = %d, not supported\n", kAdaptiveBlockScalingRuleType);
                        assert(false);
                    }

                    return reinterpret_cast<OutputType const &>(out);
                }
                else
                {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
                    asm volatile(
                        "{\n"
                        ".reg .b16 tmp0, tmp1;\n"
                        "cvt.rs.satfinite.e2m1x4.f32   tmp0, {%4, %3, %2, %1}, %9;\n"
                        "cvt.rs.satfinite.e2m1x4.f32   tmp1, {%8, %7, %6, %5}, %10;\n"
                        "mov.b32 %0, {tmp0, tmp1};\n"
                        "}"
                        : "=r"(out) : "f"(x_scaled[0]), "f"(x_scaled[1]), "f"(x_scaled[2]), "f"(x_scaled[3]),
                                      "f"(x_scaled[4]), "f"(x_scaled[5]), "f"(x_scaled[6]), "f"(x_scaled[7]),
                                      "r"(rbits), "r"(rbits));
#endif
                    return reinterpret_cast<OutputType const &>(out);
                }
            }
        }
    };

    template <bool Is_nvfp4, bool Is_2d, bool Is_4o6, bool Is_rtn, AdaptiveBlockScalingRuleType kRule, typename Engine, typename Layout, typename OutputType>
    __forceinline__ __device__ float fp4_conversion(Tensor<Engine, Layout> const &tensor, const float amax, float *sf_, OutputType *res, const uint32_t rbits)
    {
        constexpr int numel = decltype(size(tensor))::value;
        static_assert((numel == 16 && Is_nvfp4) || numel == 32);
        static_assert(std::is_same_v<OutputType, cutlass::Array<cutlass::float_e2m1_t, 8>>);

        constexpr int loop_size = 8;
        constexpr int num_loops = numel / loop_size;

        using InputType = cutlass::Array<float, loop_size>;

        Fp4ArrayQuant<Is_4o6, Is_rtn, kRule> fp4_array_quant;

        if constexpr (Is_4o6)
        {
            float err[2] = {0.0f, 0.0f};
            OutputType res_4[num_loops];
            OutputType res_6[num_loops];
            float final_err[2] = {0.0f, 0.0f};

#pragma unroll
            for (int i = 0; i < num_loops; ++i)
            {
                InputType x;
#pragma unroll
                for (int j = 0; j < loop_size; ++j)
                {
                    x[j] = static_cast<float>(tensor(i * loop_size + j));
                }
                res_4[i] = fp4_array_quant.convert(x, amax, sf_[0], rbits, &err[0]);
                res_6[i] = fp4_array_quant.convert(x, amax, sf_[1], rbits, &err[1]);
            }

            if (Is_2d){
                // For 2D tensors we want to pick the same format for the entire tensor, to keep it simple for downstream processing.
                // So we pick the format with smaller total error across the entire tensor.

                // If the method is MAE or MSE, we need to sum the error across the entire tensor. If the method is ABS_MAX, we need to take the max error across the entire tensor.
                using RedOp = std::conditional_t<kRule == AdaptiveBlockScalingRuleType::ABS_MAX_4o6, MaxOp<float>, SumOp<float>>;
                RedOp op;
                final_err[0] = Allreduce<numel>::run(err[0], op);
                final_err[1] = Allreduce<numel>::run(err[1], op);
            } else {
                final_err[0] = err[0];
                final_err[1] = err[1];
            }

            // pick_first = true means choose 4, false means choose 6
            bool const pick_first = final_err[0] < final_err[1];
#pragma unroll
            for (int i = 0; i < num_loops; ++i)
            {
                res[i] = pick_first ? res_4[i] : res_6[i];
            }
            return sf_[!pick_first];
        }
        else
        {
#pragma unroll
            for (int i = 0; i < num_loops; ++i)
            {
                InputType x;
#pragma unroll
                for (int j = 0; j < loop_size; ++j)
                {
                    x[j] = static_cast<float>(tensor(i * loop_size + j));
                }
                res[i] = fp4_array_quant.convert(x, amax, sf_[0], rbits, nullptr);
            }

            return sf_[0];
        }
    }

    __device__ __forceinline__ float clamp(float value, float min_value, float max_value)
    {
        return max(min(value, max_value), min_value);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fouroversix
