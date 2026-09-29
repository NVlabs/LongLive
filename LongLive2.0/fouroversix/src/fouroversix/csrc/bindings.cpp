#include <Python.h>
#include <torch/extension.h>

extern "C"
{
    PyObject *PyInit__C(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C", /* name of module */
            NULL, /* module documentation, may be NULL */
            -1,   /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
            NULL, /* methods */
            NULL,
            NULL,
            NULL,
            NULL,
        };
        return PyModule_Create(&module_def);
    }
}

namespace fouroversix
{
    TORCH_LIBRARY(fouroversix, m)
    {
        m.def("quantize_to_fp4(Tensor x, bool is_nvfp4, bool is_rtn, bool is_rht, bool is_2d, bool is_transpose, int selection_rule, int rbits) -> (Tensor, Tensor, Tensor)");
        m.def("gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
        m.def("gemm_mxfp4mxfp4_accum_fp32_out_bf16_tnt_sm120(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
        m.def("gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
        m.def("gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt_sm120(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
        m.def("gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
        m.def("gemm_nvfp4nvfp4_accum_fp32_out_fp16_tnt_sm120(Tensor A, Tensor B, Tensor A_sf, Tensor B_sf, Tensor alpha) -> Tensor");
    }
}