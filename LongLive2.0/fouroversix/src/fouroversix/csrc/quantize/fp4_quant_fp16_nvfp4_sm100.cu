// Splitting the different transpose modes to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "fp4_quant_launch_template.h"
namespace fouroversix {

template<>
void run_fp4_quant_<cutlass::half_t, true, false, false>(FP4_quant_params &params, cudaStream_t stream) {
    run_nvfp4_quant<cutlass::half_t, false>(params, stream);
}

} // namespace fouroversix