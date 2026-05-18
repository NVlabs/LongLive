/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/kernel_traits.h
 * Copyright (c) 2025, FourOverSix Team.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

template <bool Is_nvfp4, typename elem_type = cutlass::half_t>
struct Base_kernel_traits
{

    static constexpr float E2M1_MAX_VALUE = 6.0f;
    static constexpr float E4M3_MIN_VALUE = -448.0f;
    static constexpr float E4M3_MAX_VALUE = 448.0f;
    static constexpr float E4M3_MAX_FOUROVERSIX = 256.0f;

    using Element = elem_type;
    using ScaleFactor = std::conditional_t<Is_nvfp4, cutlass::float_e4m3_t, uint8_t>;
    // using ElementXe2m1Packed = std::conditional_t<Is_nvfp4, uint64_t, uint128_t>;
    using NormConst = float;
    static constexpr bool Has_cp_async = true;

    using index_t = int64_t;

    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template <int kBlockM_, int kBlockN_, int kNWarps_, bool Is_nvfp4, bool Is_transpose, typename elem_type = cutlass::half_t, typename Base = Base_kernel_traits<Is_nvfp4, elem_type>>
struct FP4_quant_kernel_traits : public Base
{
    static constexpr float E2M1_MAX_VALUE = Base::E2M1_MAX_VALUE;
    static constexpr float E2M1_MAX_FOUR = 4;
    static constexpr float E4M3_MIN_VALUE = Base::E4M3_MIN_VALUE;
    static constexpr float E4M3_MAX_VALUE = Base::E4M3_MAX_VALUE;
    static constexpr float E4M3_MAX_FOUROVERSIX = Base::E4M3_MAX_FOUROVERSIX;

    using Element = typename Base::Element;
    using ScaleFactor = typename Base::ScaleFactor;
    using NormConst = typename Base::NormConst;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kGroupN = Is_nvfp4 ? 16 : 32; // 16 or 32 elements
    static_assert(kBlockM % 128 == 0);
    static_assert(kBlockN % (kGroupN * 4) == 0);
    static_assert(kBlockN % 64 == 0);
    static constexpr int kBlockNSmem = 64; // each cache line is 128 bytes, so we need to align to 64 bytes
    static constexpr int kBlockNGmem = kBlockN % 128 == 0 ? 128 : 64;

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>, _1, _1>>, // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _16>>;

    // static constexpr int kGroupN = Is_nvfp4 ? 16 : 32; // 16 or 32 elements
    // static constexpr int kSwizzleM = Is_nvfp4 ? 4 : 5; // 16 or 32 elements
    // static constexpr int kSwizzleS = Is_nvfp4 ? 2 : 1; // 4 or 2 elements
    // static constexpr int kSwizzleB = Is_nvfp4 ? 2 : 1; // 2 or 1 bits

    static constexpr int kNumGroupsInRow = kBlockN / kGroupN;
    static_assert(kBlockM % kGroupN == 0, "kBlockM must be a multiple of kGroupN if is 2d");
    static constexpr int kNumGroupsInCol = kBlockM; // for 2d scale factor
    // static constexpr int kSwizzleM = 3;
    // static constexpr int kSwizzleS = 3;
    // static constexpr int kSwizzleB = 2;

    // using SmemLayoutAtomX = decltype(
    //     composition(Swizzle<kSwizzleB, kSwizzleM, kSwizzleS>{},
    //                 // This has to be kBlockNSmem, using kHeadDim gives wrong results for d=128
    //                 Layout<Shape<_8, Int<kBlockNSmem>>,
    //                         Stride<Int<kBlockNSmem>, _1>>{}));
    // using SmemLayoutX = decltype(tile_to_shape(
    //     SmemLayoutAtomX{},
    //     Shape<Int<kBlockM>, Int<kBlockN>>{}));
    using SmemLayoutX = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;

    using SmemLayoutXTransposed = decltype(composition(SmemLayoutX{}, make_layout(Shape<Int<kBlockN>, Int<kBlockM>>{}, GenRowMajor{})));
    using SmemLayoutXTransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutXTransposed{}));

    using SmemLayoutSFT = Layout<Shape<Int<kBlockM>, Int<kBlockN / kGroupN>>,
                                 Stride<Int<kBlockN / kGroupN>, _1>>;

    static constexpr int kBlockMSF = kBlockM / 128 * 32 * int(kBlockN / (kGroupN * 4));
    static constexpr int kBlockNSF = 16;
    using SmemLayoutSF = Layout<Shape<Int<kBlockMSF>, Int<kBlockNSF>>,
                                Stride<Int<kBlockNSF>, _1>>;

    using SmemLayout = SmemLayoutX;
    static constexpr int kSmemXSize = size(SmemLayout{}) * sizeof(Element);
    static constexpr int kSmemXe2m1Size = kSmemXSize / 4;
    static constexpr int kSmemSFTSize = size(SmemLayoutSFT{}) * sizeof(float);
    static constexpr int kSmemSFSize = size(SmemLayoutSF{}) * sizeof(ScaleFactor);
    static constexpr int kSmemSize = kSmemXSize + kSmemXe2m1Size + kSmemSFTSize + kSmemSFSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kBlockN % kGmemElemsPerLoad == 0, "kBlockN must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kBlockNSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtomX = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                   Stride<Int<kGmemThreadsPerRow>, _1>>;

    using Gmem_copy_atom_x = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;

    using GmemTiledCopyX = decltype(make_tiled_copy(Gmem_copy_atom_x{},
                                                    GmemLayoutAtomX{},
                                                    Layout<Shape<_1, _8>>{}));

    using Gmem_copy_atom_sft = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<64>, float>;

    using GmemLayoutAtomSFT = Layout<
        Shape<_64, _1>,
        Stride<_1, _0>>;

    using GmemTiledCopySFT = decltype(make_tiled_copy(Gmem_copy_atom_sft{}, GmemLayoutAtomSFT{}, Layout<Shape<_1, _4>>{}));
};

////////////////////////////////////////////////////////////////////////////////////////////////////
