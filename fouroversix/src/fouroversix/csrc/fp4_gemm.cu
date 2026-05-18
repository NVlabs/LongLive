#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"

#include "helper.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "element_traits.hpp"

namespace fouroversix
{
    using namespace cute;

    // Adapted from example 72b
    template <typename ElementA,
              typename MmaTileShape = Shape<_128, _128, _256>,
              typename ClusterShape = Shape<_2, _4, _1>,
              typename KernelMainloopPolicy = cutlass::gemm::collective::KernelScheduleAuto,
              int AlignmentA = 32,
              typename ElementB = ElementA,
              int AlignmentB = 32,
              typename LayoutATag = cutlass::layout::RowMajor,
              typename LayoutBTag = cutlass::layout::ColumnMajor,
              typename ElementD = cutlass::bfloat16_t,
              typename ArchTag = cutlass::arch::Sm100>
    torch::Tensor gemm_fp4fp4_accum_fp32(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor const &A_sf, torch::Tensor const &B_sf, torch::Tensor const &alpha)
    {
        at::cuda::CUDAGuard device_guard(A.device());

        // C/D matrix configuration
        using ElementC = void;                        // Element type for C matrix operand
        using LayoutCTag = cutlass::layout::RowMajor; // Layout type for C matrix operand
        using LayoutDTag = cutlass::layout::RowMajor; // Layout type for D matrix operand

        constexpr int AlignmentC = 1;                                           // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
        constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value; // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)

        // Kernel functional config
        using ElementAccumulator = float;                                // Element type for internal accumulation
        using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp; // Operator class tag

        using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
            ArchTag, OperatorClass,
            MmaTileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccumulator, ElementAccumulator,
            ElementC, LayoutCTag, AlignmentC,
            ElementD, LayoutDTag, AlignmentD,
            cutlass::epilogue::collective::EpilogueScheduleAuto // Epilogue schedule policy
            >::CollectiveOp;

        using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag, OperatorClass,
            ElementA, LayoutATag, AlignmentA,
            ElementB, LayoutBTag, AlignmentB,
            ElementAccumulator,
            MmaTileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            KernelMainloopPolicy>::CollectiveOp;

        using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
            Shape<int, int, int, int>, // Indicates ProblemShape
            CollectiveMainloop,
            CollectiveEpilogue,
            void>;

        using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

        // Reference device GEMM implementation type
        using StrideA = typename Gemm::GemmKernel::StrideA;
        using StrideB = typename Gemm::GemmKernel::StrideB;
        using StrideC = typename Gemm::GemmKernel::StrideC;
        using StrideD = typename Gemm::GemmKernel::StrideD;

        using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA; // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
        using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB; // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.

        using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

        torch::checkAllContiguous("gemm_fp4fp4_accum_fp32_out_bf16", {{A, "A", 0}, {B, "B", 1}, {A_sf, "A_sf", 2}, {B_sf, "B_sf", 3}, {alpha, "alpha", 4}});
        torch::checkDeviceType("gemm_fp4fp4_accum_fp32_out_bf16", {A, B, A_sf, B_sf, alpha}, at::DeviceType::CUDA);
        torch::checkAllSameGPU("gemm_fp4fp4_accum_fp32_out_bf16", {{A, "A", 0}, {B, "B", 1}, {A_sf, "A_sf", 2}, {B_sf, "B_sf", 3}, {alpha, "alpha", 4}});

        check_block_scale_factor_type<ElementA>(A_sf, "A_sf");
        check_block_scale_factor_type<ElementB>(B_sf, "B_sf");

        auto [M, N, K] = check_and_get_fp4_matmul_dims<ElementA, LayoutATag, ElementB, LayoutBTag>(A, B, A_sf, B_sf);
        auto D = torch::empty({M, N}, torch::dtype(element_traits<ElementD>::scalar_type).device(A.device()));

        Gemm gemm;

        // Create stride and layout information for the packed tensors
        // For packed NVFP4 tensors, we need to use the appropriate stride and layout
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

        // Create scale factor layouts
        LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
        LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {// Mainloop arguments
             static_cast<typename ElementA::DataType const *>(A.data_ptr()), stride_A,
             static_cast<typename ElementB::DataType const *>(B.data_ptr()), stride_B,
             static_cast<typename ElementA::ScaleFactorType const *>(A_sf.data_ptr()), layout_SFA,
             static_cast<typename ElementB::ScaleFactorType const *>(B_sf.data_ptr()), layout_SFB},
            {// Epilogue arguments
             {1.0f, 0.0f},
             nullptr,
             stride_C,
             static_cast<ElementD *>(D.data_ptr()),
             stride_D}};

        args.epilogue.thread.alpha_ptr = static_cast<ElementAccumulator *>(alpha.data_ptr());

        // Check if the problem size is supported or not
        CUTLASS_CHECK(gemm.can_implement(args));

        // Initialize CUTLASS kernel with arguments and workspace pointer
        CUTLASS_CHECK(gemm.initialize(args));

        CUTLASS_CHECK(gemm.run(args));

        return D;
    }

    TORCH_LIBRARY_IMPL(fouroversix, CUDA, m)
    {
        /* MXFP4 */
        m.impl("gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt",
               &gemm_fp4fp4_accum_fp32<
                   cutlass::mx_float4_t<cutlass::float_e2m1_t>,
                   Shape<_256, _256, _256>,
                   Shape<_4, _1, _1>,
                   cutlass::gemm::KernelTmaWarpSpecialized2SmMxf4Sm100>);

        /* NVFP4 */
        m.impl("gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt",
               &gemm_fp4fp4_accum_fp32<
                   cutlass::nv_float4_t<cutlass::float_e2m1_t>,
                   Shape<_256, _256, _256>,
                   Shape<_4, _1, _1>,
                   cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100>);

        m.impl("gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt",
               &gemm_fp4fp4_accum_fp32<
                   cutlass::nv_float4_t<cutlass::float_e2m1_t>,
                   Shape<_256, _256, _256>,
                   Shape<_4, _1, _1>,
                   cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100,
                   32,
                   cutlass::nv_float4_t<cutlass::float_e2m1_t>,
                   32,
                   cutlass::layout::RowMajor,
                   cutlass::layout::ColumnMajor,
                   cutlass::half_t>);
    }
}