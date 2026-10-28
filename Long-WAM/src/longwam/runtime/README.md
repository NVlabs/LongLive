<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/README.md
License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.
-->

# Runtime documentation

See [infra/README.md](../../../infra/README.md) for installation, device builds,
async/streaming inference and robot interfaces. Python code remains in `longwam.runtime`.

The spawned inference transport allows one request in flight. A response timeout
keeps that request pending, so the same future can be awaited again. If the
outgoing observation queue is full, a rejected request can be retried after the
backlog drains. Sender write or serialization failures are terminal: pending
futures raise `RemoteInferenceError`, and the worker must be closed and recreated.
Cleanup also handles this failed-transport state without retrying a graceful
shutdown over the broken sender.
