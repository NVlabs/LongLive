#include "cutlass/bfloat16.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include <ATen/ATen.h>
#include <type_traits>

namespace fouroversix
{
    template <typename T>
    struct element_traits;

    template <>
    struct element_traits<cutlass::mx_float4_t<cutlass::float_e2m1_t>>
    {
        static constexpr size_t block_size = 32;
        static constexpr size_t packing_factor = 2;
        static constexpr size_t packed_bytes = 1;
        static constexpr at::ScalarType block_scale_factor_type = at::kFloat8_e8m0fnu;
        static constexpr at::ScalarType scalar_type = at::kByte;
    };

    template <>
    struct element_traits<cutlass::mx_float6_t<cutlass::float_e2m3_t>>
    {
        static constexpr size_t block_size = 32;
        static constexpr size_t packing_factor = 4;
        static constexpr size_t packed_bytes = 3;
        static constexpr at::ScalarType block_scale_factor_type = at::kFloat8_e8m0fnu;
        static constexpr at::ScalarType scalar_type = at::kByte;
    };

    template <>
    struct element_traits<cutlass::mx_float8_t<cutlass::float_e5m2_t>>
    {
        static constexpr size_t block_size = 32;
        static constexpr size_t packing_factor = 1;
        static constexpr size_t packed_bytes = 1;
        static constexpr at::ScalarType block_scale_factor_type = at::kFloat8_e8m0fnu;
        static constexpr at::ScalarType scalar_type = at::kByte;
    };

    template <>
    struct element_traits<cutlass::nv_float4_t<cutlass::float_e2m1_t>>
    {
        static constexpr size_t block_size = 16;
        static constexpr size_t packing_factor = 2;
        static constexpr size_t packed_bytes = 1;
        static constexpr at::ScalarType block_scale_factor_type = at::kFloat8_e4m3fn;
        static constexpr at::ScalarType scalar_type = at::kByte;
    };

    template <>
    struct element_traits<cutlass::bfloat16_t>
    {
        static constexpr at::ScalarType scalar_type = at::kBFloat16;
    };

    template <>
    struct element_traits<cutlass::half_t>
    {
        static constexpr at::ScalarType scalar_type = at::kHalf;
    };

    template <>
    struct element_traits<float>
    {
        static constexpr at::ScalarType scalar_type = at::kFloat;
    };

    template <typename Element>
    void check_block_scale_factor_type(const at::Tensor &t, const char *name)
    {
        TORCH_CHECK(
            t.scalar_type() == element_traits<Element>::block_scale_factor_type,
            name, " must be ", at::toString(element_traits<Element>::block_scale_factor_type));
    }

    template <typename ElementA, typename LayoutATag, typename ElementB, typename LayoutBTag>
    std::tuple<int, int, int>
    check_and_get_bf16_matmul_dims(const at::Tensor &A, const at::Tensor &B)
    {
        TORCH_CHECK(A.scalar_type() == at::kBFloat16, "A must be bfloat16");
        TORCH_CHECK(B.scalar_type() == at::kBFloat16, "B must be bfloat16");
        TORCH_CHECK(A.dim() == 2, "A must be 2D");
        TORCH_CHECK(B.dim() == 2, "B must be 2D");

        // Unpack sizes
        int a_rows = A.size(0), a_cols = A.size(1);
        int b_rows = B.size(0), b_cols = B.size(1);

        // Layout-based interpretation
        int M, K_A;
        if constexpr (std::is_same_v<LayoutATag, cutlass::layout::RowMajor>)
        {
            M = a_rows;
            K_A = a_cols;
        }
        else if constexpr (std::is_same_v<LayoutATag, cutlass::layout::ColumnMajor>)
        {
            M = a_cols;
            K_A = a_rows;
        }

        int N, K_B;
        if constexpr (std::is_same_v<LayoutBTag, cutlass::layout::RowMajor>)
        {
            K_B = b_rows;
            N = b_cols;
        }
        else if constexpr (std::is_same_v<LayoutBTag, cutlass::layout::ColumnMajor>)
        {
            K_B = b_cols;
            N = b_rows;
        }

        TORCH_CHECK(K_A == K_B, "Inner dims mismatch: ", K_A, " vs ", K_B);

        return {M, N, K_A};
    }

    template <typename ElementA, typename LayoutATag, typename ElementB, typename LayoutBTag>
    std::tuple<int, int, int>
    check_and_get_fp4_matmul_dims(const at::Tensor &A, const at::Tensor &B,
                                  const at::Tensor &A_sf, const at::Tensor &B_sf)
    {
        TORCH_CHECK(A.scalar_type() == at::kByte, "A must be uint8");
        TORCH_CHECK(B.scalar_type() == at::kByte, "B must be uint8");
        TORCH_CHECK(A.dim() == 2, "A must be 2D");
        TORCH_CHECK(B.dim() == 2, "B must be 2D");
        TORCH_CHECK(A.size(1) >= 16, "A K-dim must be >= 16");
        TORCH_CHECK(B.size(1) >= 16, "B K-dim must be >= 16");

        // Unpack 4-bit logical sizes
        int a_rows = A.size(0), a_cols = A.size(1) * element_traits<ElementA>::packing_factor / element_traits<ElementA>::packed_bytes;
        int b_rows = B.size(0), b_cols = B.size(1) * element_traits<ElementB>::packing_factor / element_traits<ElementB>::packed_bytes;

        // Layout-based interpretation
        int M, K_A;
        if constexpr (std::is_same_v<LayoutATag, cutlass::layout::RowMajor>)
        {
            M = a_rows;
            K_A = a_cols;
        }
        else if constexpr (std::is_same_v<LayoutATag, cutlass::layout::ColumnMajor>)
        {
            M = a_cols;
            K_A = a_rows;
        }

        int N, K_B;
        if constexpr (std::is_same_v<LayoutBTag, cutlass::layout::RowMajor>)
        {
            K_B = b_rows;
            N = b_cols;
        }
        else if constexpr (std::is_same_v<LayoutBTag, cutlass::layout::ColumnMajor>)
        {
            K_B = b_cols;
            N = b_rows;
        }

        TORCH_CHECK(K_A == K_B, "Inner dims mismatch: ", K_A, " vs ", K_B);

        // Scale factor checks
        TORCH_CHECK(A_sf.numel() * element_traits<ElementA>::block_size == size_t(a_rows) * size_t(a_cols), "A_sf size mismatch");
        TORCH_CHECK(B_sf.numel() * element_traits<ElementB>::block_size == size_t(b_rows) * size_t(b_cols), "B_sf size mismatch");

        return {M, N, K_A};
    }
}