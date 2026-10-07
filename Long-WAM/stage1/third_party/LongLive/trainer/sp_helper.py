# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: The LongLive contributors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/NVlabs/LongLive @ 0308b126accba9440b8caa45bcf7bec0877933e1 :: trainer/sp_helper.py
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/trainer/sp_helper.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for LongLive SP training helpers."""

from wan_5b.distributed.sp_training import (
    DEFAULT_SP_VAE_HALO_LATENTS,
    SequenceParallelHelper,
)

__all__ = ["DEFAULT_SP_VAE_HALO_LATENTS", "SequenceParallelHelper"]
