# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LongLive-H3 generator worker (few-step chunked-AR distillation of MiniMax-H3, video + audio).

* :mod:`.core`          the exact KV-cached sampler, VAE decoding and transformer loading (GPU);
* :mod:`.text_encoder`  the shared MiniMax-H3 text / picture encoder service (one GPU for all workers);
* :mod:`.worker`        one worker per GPU serving the job queue of :mod:`agentic_longlive.workers.protocol`.

Setup and launch: ``scripts/workers/setup_longlive_h3.sh`` and ``scripts/workers/serve_longlive_h3.sh``
(``docs/generators.md``). The reference worker here renders one shot per GPU; it is not the optimized real-time
runtime.
"""
