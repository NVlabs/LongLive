<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Long-WAM RTX 4090 contributors
SPDX-License-Identifier: Apache-2.0
Provenance: Original opt-in RTX4090 Fast mode documentation.
Source: https://github.com/NVLabs/LongLive/pull/118
-->

# Experimental RTX4090 Fast mode

Fast is an explicitly selected alternative to the default Strict RTX4090 mode.
It retains BF16/FP32 parameter dtypes, unmerged adapters, fresh observation VAE
and the original inference step counts. It changes the causal clean-prefix
execution path and floating-point rounding. Actions and closed-loop outcomes
can differ; numerical equivalence and capability noninferiority are not claimed.

## Selection and tested scope

Load the complete checkpoint and adapters into a fresh evaluation model first:

```python
from longwam.runtime.backends import prepare_backend

model.eval().requires_grad_(False)
state = prepare_backend(model, "rtx4090", {"rtx4090_mode": "fast"})
print(state.to_dict())
```

Omitting the override keeps Strict. A prepared model cannot switch modes;
create a fresh model instead. Neither mode requires a FourOverSix build.

The CLI accepts the same selection. This planning example uses no model or GPU:

```bash
longwam eval libero --dry-run hardware=rtx4090 action_horizon=32 \
  execute_steps=10 replan_steps=10 num_inference_steps=10 seed=1 \
  optimization.rtx4090_mode=fast
```

The tested contract is Linux RTX4090/SM8.9, synchronous LIBERO-IDM, P48,
13 observation frames, batch1, H32, V4/A10, CFG1 and CPU-seeded noise.
The frame-local causal video expert, shared expert identity and unquantized KV
are checked before installing Fast. Existing shared SM89 call guards remain.
Async/streaming VAE, training, other shapes/checkpoints and other GPUs are not
validated by these results. Offline LoRA compatibility is not task validation
of a trained adapter.

## Measurements

Same 24564MiB RTX4090 at its pre-existing 480 W limit, driver 595.91.07,
PyTorch 2.7.1/cu128, Triton 3.3.1, NumPy 2.2.6 and Pillow 13.0.0.
Six blocks cover all three-backend orders; each backend uses a fresh process,
32 warmups and 32 measurements on 16 recorded inputs with seeds 1 and 2.

| Metric | Reference | Strict | Fast |
|---|---:|---:|---:|
| Mean of process median full-model latency |911.415ms|371.826ms|309.527ms|
| Mean of process median request latency |961.232ms|421.741ms|359.499ms|

Fast gives 2.945x full-model and 2.674x request speedup over Reference.
The incremental request-latency reduction versus Strict is 14.76%.
Block-bootstrap lower bounds are 2.914x and 2.648x respectively; six matched
blocks, 50000 draws, seed 20261023, one-sided alpha 0.05/3 per endpoint.
Full-model includes fresh VAE, denoising and CPU action return; request adds
the same image/state preparation. Loading, hashing, capture/warmup, initial
text encoding, simulation and imagined-RGB decoding are excluded.
These are warm inference results, not whole-task speedups or default-recipe scores.

## Native outcome evidence and limits

At the user-requested 850-pair stop, 34 tasks have 25 states each, fixed seed 1.
Strict succeeds 840/850 (98.82%); Fast 837/850 (98.47%). Both succeed 834,
both fail 7, Strict alone succeeds 6 and Fast alone succeeds 3.
Fast-minus-Strict is -0.353 percentage points; exact two-sided McNemar p=0.5078125.
This did not detect a significant difference under the test assumptions;
it does not establish equivalence or exclude degradation.
The conservative independent-pair 95% interval is [-1.608, +0.920] percentage
points; task dependence and uncovered tasks limit inference.

The six uncovered tasks are goal05, object02/04/05/07 and spatial08.
The original 1000-case and 400-seed plan was stopped, not completed or passed.
The original conservative loss-count gate is not met: 6 observed losses exceed
the original maximum 3 at n=1000. The study did not meet that criterion.

All nine discordant pairs have matching recorded initial conditions, intact
raw and internally consistent native success flags and executed actions.
Their first actual policy inputs match, but first policy outputs differ before
the first post-action RGB divergence. This rules out a rendering-only account
of the first recorded divergence; it does not establish each failure's causal
mechanism or quantify Strict-versus-Strict variability.

In six of these nine selected pairs, the successful side's native predicate
does not persist throughout a supplemental 2 s zero-EEF hold. This occurs in
both modes. The official first-success score remains unchanged; the hold is
not continued policy behavior and predicate loss is not a drop-event label.

Anonymous screening covered 220 pairs, not a full event assessment.
Twenty overall impressions were archived; the reviewer also reports viewing
several dozen additional pairs, mostly hard to distinguish visually, without
a complete per-case label log. Eight selected suspects were unblinded after
answers were saved (Fast preferred 3, Strict 2, similar 3). Selection prevents
population-rate inference. Formal jitter/misgrasp-or-regrasp/drop labels are
incomplete, and future labels for those eight must be marked post-unblinding.
No added-event-rate upper bound is claimed.

## Tests and reproduction

Existing focused CPU admission/dispatch tests, GPU cache/parameter regressions,
same-input replay and a 60-minute resident development soak are retained.
The soak verified 9376 operational outputs and stable capture/workspace limits;
it is not native capability validation. Base and adapter hashes are unchanged.

The companion local evidence pack includes the exact source manifest, recorded
fixtures, immutable original inference harness, a CPU result summarizer,
timing design, raw timing metadata and source/runner identities.
It contains no large weights or private adapters. A portable three-process
smoke validates packaging only; it does not replace the six-block timing study.
The evidence attachment has not been published and no hosted CI is claimed.
