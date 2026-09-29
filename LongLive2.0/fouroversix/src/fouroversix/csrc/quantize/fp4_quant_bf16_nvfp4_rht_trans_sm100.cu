// Splitting the different transpose modes to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "fp4_quant_launch_template.h"
namespace fouroversix {

template<>
void run_fp4_quant_<cutlass::bfloat16_t, true, true, true>(FP4_quant_params &params, cudaStream_t stream) {
    run_nvfp4_quant_rht<cutlass::bfloat16_t, true>(params, stream);
}

} // namespace fouroversix