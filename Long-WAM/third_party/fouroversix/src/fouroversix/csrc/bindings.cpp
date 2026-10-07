// Long-WAM file attribution; existing upstream notices are retained below.
// SPDX-FileCopyrightText: Copyright (c) 2025 Jack Cook
// SPDX-License-Identifier: MIT
// Provenance: Copied from the source snapshot listed below.
// Source: https://github.com/mit-han-lab/fouroversix @ 4c0db4bf3027272c61e32ca7e2cdb9c7fa5bcecb :: src/fouroversix/csrc/bindings.cpp
// Changes: Source is the vendored FourOverSix snapshot; upstream-file notices remain below.
// License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
// Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: third_party/fouroversix/src/fouroversix/csrc/bindings.cpp
// End Long-WAM attribution.

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
