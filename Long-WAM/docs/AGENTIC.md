<!--
Long-WAM file attribution; existing upstream notices are retained below.
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
Provenance: Original Long-WAM implementation.
License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
End Long-WAM attribution.
-->

# Agentic API

Agentic API is Long-WAM's benchmark-independent model/tool runtime. The current
backend is **Codex app-server**; the released environment adapter is
**RoboCasa365**. Model choice and benchmark adaptation are separate concerns.
Other benchmarks can implement the small adapter interface below; they are not
automatically enabled by changing the benchmark name.

## Quick start

Install `pip install -e '.[agentic]'` alongside the RoboCasa simulator requirements.
Authenticate using your own supported Codex login; no credentials are bundled.
The host follows the [Codex app-server protocol](https://learn.chatgpt.com/docs/app-server).
The recorded experiment used CLI 0.154.0. Verify your installed CLI's experimental
dynamic-tool/permission protocol and account access before a real run.

### Your own authentication

Use `codex login` with your own account, or pipe your own `OPENAI_API_KEY` to
`codex login --with-api-key`. Follow the [official authentication guide](https://learn.chatgpt.com/docs/auth).
Keep login state outside the repository (the default is `~/.codex`).
No API-key field, account ID, token or credential file is embedded in our YAMLs;
authentication remains with the local Codex client.

Never put keys in `key=value` overrides, prompts, screenshots or audit outputs.
Do not commit `.env`, `auth.json`, private keys or your Codex home. The
agent's shell tools receive a minimal environment, not the host's API-key
variables; their HOME points to the episode workspace. Audit logs redact
recognized credential fields/token formats. Use a dedicated account/environment
and review the CLI sandbox; these controls are not a full isolation guarantee.
Only trusted local configs, checkpoints and adapters should be used. Simulator
policy sockets are local and owner-only, not an authenticated public API. Keep
robot bridges on a private network; do not expose them directly to the Internet.

```bash
# CPU-only planning: no simulator, GPU model, Codex process or paid model call.
bash scripts/robocasa365/eval.sh --agentic --dry-run

# One episode first. Use a fresh output directory for every run.
bash scripts/robocasa365/eval.sh --agentic \
  checkpoint=/path/to/model.pt model_config=/path/to/config.yaml \
  stats=/path/to/dataset_stats.json \
  simulator_python=/path/to/robocasa-env/bin/python \
  agentic.codex=/path/to/codex agentic.allow_model_upload=true \
  agentic.max_total_tokens=100000 \
  'tasks=[PrepareCoffee]' episodes=1 output_dir=outputs/robocasa365-agentic
```

The recipe is [robocasa365_agentic.yaml](../configs/eval/robocasa365_agentic.yaml).
Its defaults are `agentic.model=gpt-6-astra` and `agentic.effort=xhigh`.
Both are parameters, not a model allowlist:

```bash
bash scripts/robocasa365/eval.sh --agentic --dry-run \
  agentic.model=YOUR_EXACT_CODEX_MODEL_ID agentic.effort=high
```

For an Astra, Sol or another model, pass the **exact model ID available to your
Codex account** and a supported effort. No shorthand-to-version mapping or
automatic model fallback is performed. Actual model/effort and CLI version are
recorded with each episode. Changing the model is a new experiment, not an exact
reproduction of the default setting. This interface does not yet implement
other providers or the Responses API.

## Architecture and extension

```text
src/longwam/
├── agentic/                      # Reusable runtime; no benchmark imports
│   ├── api.py                    # AgentSpec, AgenticAdapter, AgenticEpisode
│   ├── config.py                 # Model, effort, consent, token budget, timeout
│   ├── codex.py                  # Sessions, tool calls, recovery and audit logs
│   └── _vendor/                  # Shared transport/image helpers
└── benchmarks/robocasa/agentic/   # RoboCasa365 adapter
    ├── adapter.py                # Prompts and generic-API tool binding
    ├── session.py                # Simulator and Long-WAM policy connection
    ├── tools.py                  # Observation → proposal → bounded execution
    ├── contract.py               # Robot/action conventions and safety limits
    ├── skill/                    # Benchmark-specific prompt assets
    ├── evaluate.py              # Task/episode aggregation
    └── run_rollout.py            # One episode
```

To add a benchmark, implement the protocols in
[api.py](../src/longwam/agentic/api.py):

1. **AgenticAdapter.prepare(audit)** creates a workspace below the fresh audit
   directory and returns `AgentSpec`: instructions, tool schemas, initial
   message, image-bearing tool names and benchmark metadata.
2. **AgenticEpisode** provides `finished`, `step_id`, `next_call()`,
   `call_tool(name, arguments)` and `rejected_input(error, count)`.
   It owns the simulator, policy inference, action validation and native success
   checks. The runtime never assumes camera count, robot type or action size.
3. Connect them using the public Python API:

```python
from pathlib import Path
from longwam.agentic import AgenticConfig, CodexAgent

# adapter and episode are your implementations of the two protocols.
settings = AgenticConfig(
    model="gpt-6-astra", effort="xhigh",
    allow_model_upload=True, max_total_tokens=100000,
)
with CodexAgent(Path("outputs/my-benchmark/episode-000/agentic"),
                adapter=adapter, config=settings) as agent:
    agent.run(episode)
```

Only raise `longwam.agentic.InputError` when rejecting arguments **before any
action/inference with side effects**; the model may correct and retry that call.
Other exceptions abort the episode. Do not replay actions, silently reset an
environment, or treat a model's success claim as native success. The caller owns
simulator cleanup and incomplete-result recording; the context manager closes
Codex. The RoboCasa [adapter](../src/longwam/benchmarks/robocasa/agentic/adapter.py)
and [episode runner](../src/longwam/benchmarks/robocasa/agentic/run_rollout.py)
show the complete integration.

## RoboCasa365 protocol

Only the selected **hybrid_decompose v1** recipe is released. The agent gives
Long-WAM one atomic sub-instruction and reviews the 32×12 action proposal.
It may execute 1–15 student steps or 1–5 bounded EEF correction steps; after
three consecutive corrections it must return control to the student.
PandaOmron runs at 20 Hz, with the same P48 (2.4 s) observation-history policy.
These constraints and the benchmark instructions remain in the RoboCasa adapter,
not the reusable runtime. Prompt identity wording is model-neutral; the action,
gate and native-termination rules are unchanged.

The default protocol is Target50, pretrain split, environment seed 7, five
episodes/task, using the same student checkpoint as standalone evaluation.
That is **250 episodes**. For a matched student-only comparison explicitly use
`episodes=5`; standalone evaluation defaults to 50 episodes/task.
This is an inference experiment, not a student training stage.

## Data sharing, budgets and failures

Real runs require explicit observation-upload consent and a positive per-episode
token budget. Simulator images, robot state, task text, action proposals and
same-episode history are sent to OpenAI. Use only data you are authorized to share.
The example budget is adjustable, not a reproduction hyperparameter or price
estimate. Asynchronous usage reports can overshoot; it is **not a billing cap**.

Timeouts, model/configuration errors and budget cutoffs fail the episode; they
are never counted as a completed benchmark result. Network recovery continues
the same thread without resetting the simulator or replaying executed actions.
The host enables shell/image tools inside an episode workspace. Use a dedicated,
credential-minimal environment and review your CLI's permission profile.

Dry-run performs no model calls. Prompts, tool schema, model/version, token usage
and tool decisions are written only inside the chosen ignored output directory.

## Compatibility and tests

Existing `--gpt6`, `gpt6.*`, `.[gpt6]`, the old YAML filename and legacy Python
imports remain compatibility entry points; new code should use Agentic API.
Do not specify the same setting through both old and new namespaces.
The legacy Python `CodexPolicy` constructor also requires explicit
`allow_model_upload=True` before opening a model session.

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider \
  tests/test_agentic.py tests/test_public_variants.py tests/test_license_headers.py
```

Tests use an in-memory Codex transport and synthetic episodes, including a
non-RoboCasa adapter. They do not certify account access, live simulator rollout
quality or real-model performance.
