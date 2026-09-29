<!--
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0

No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.

SPDX-License-Identifier: Apache-2.0
-->

# LongLive KV Dequant CUDA Extension

Build from this directory:

```bash
cd utils/kernel
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=4 \
  python setup.py build_ext --inplace
```

Runtime import:

```python
from utils.kernel.kv_dequant import dequantize_kv_cache_fp4
```

`utils.quant.dequantize_kv_cache()` already calls this extension first and falls
back to the original Triton path if the extension is not built.

For direct calls, pass the same scale limits used by the QuantizedTensor's
`scale_rule`:

- `static_6`: `e2m1_max=6.0`, `e4m3_max=448.0`
- `static_4`: `e2m1_max=4.0`, `e4m3_max=448.0`
- `mse` / `l1_norm` / `abs_max` 4o6 modes: `e2m1_max=6.0`, `e4m3_max=256.0`

The normal `utils.quant.dequantize_kv_cache()` path reads these values from
`qt.scale_rule`, so manual selection is not needed there.

You can also pass `scale_rule` directly:

```python
out = dequantize_kv_cache_fp4(
    values,
    scale_factors,
    amax,
    num_heads=num_heads,
    block_token_size=block_token_size,
    dtype=torch.bfloat16,
    scale_rule="static_6",
)
```
