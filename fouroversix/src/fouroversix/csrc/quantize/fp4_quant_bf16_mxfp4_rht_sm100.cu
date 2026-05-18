// Splitting the different transpose modes to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "fp4_quant_launch_template.h"
namespace fouroversix {

template<>
void run_fp4_quant_<cutlass::bfloat16_t, false, true, false>(FP4_quant_params &params, cudaStream_t stream) {
    run_mxfp4_quant_rht<cutlass::bfloat16_t, false>(params, stream);
}

} // namespace fouroversix