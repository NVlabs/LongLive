# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# End Long-WAM attribution.

"""Small, dependency-free helpers for LIBERO policy-call latency reporting."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for an already sorted sequence."""
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize_policy_call_latency(
    raw_seconds: Sequence[float],
    *,
    warmup_calls: int = 1,
) -> dict[str, object]:
    """Summarize policy-call latency while retaining all raw measurements.

    The first ``warmup_calls`` values are excluded only from aggregate statistics.
    They remain in ``raw_seconds`` so a later analysis can choose another warmup rule.
    """
    if warmup_calls < 0:
        raise ValueError(f"warmup_calls must be non-negative, got {warmup_calls}")

    raw = [float(value) for value in raw_seconds]
    if any(not math.isfinite(value) or value < 0.0 for value in raw):
        raise ValueError("Policy-call latencies must be finite, non-negative seconds.")

    excluded = min(warmup_calls, len(raw))
    measured = sorted(raw[excluded:])
    summary: dict[str, object] = {
        "unit": "seconds_per_policy_call",
        "count": len(raw),
        "warmup_calls_requested": warmup_calls,
        "warmup_calls_excluded": excluded,
        "measured_count": len(measured),
        "raw_seconds": raw,
        "mean_seconds": None,
        "p50_seconds": None,
        "p95_seconds": None,
        "max_seconds": None,
    }
    if measured:
        summary.update(
            {
                "mean_seconds": float(sum(measured) / len(measured)),
                "p50_seconds": _percentile(measured, 0.50),
                "p95_seconds": _percentile(measured, 0.95),
                "max_seconds": float(measured[-1]),
            }
        )
    return summary
