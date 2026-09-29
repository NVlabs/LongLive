# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.distributed as dist


def _is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _log_once(message: str) -> None:
    if _is_main_process():
        print(message)


class SafeCompiledCallable:
    """Lazy torch.compile wrapper that falls back to eager on compile/runtime errors."""

    def __init__(
        self,
        fn,
        *,
        name: str,
        backend: str = "inductor",
        mode: str | None = "max-autotune-no-cudagraphs",
        fullgraph: bool = False,
        dynamic: bool | None = False,
        options: dict | None = None,
        suppress_errors: bool = True,
    ) -> None:
        self.fn = fn
        self.name = name
        self.enabled = True
        self.failed = False
        self.failure_reason = None

        if suppress_errors:
            try:
                import torch._dynamo as torch_dynamo

                torch_dynamo.config.suppress_errors = True
            except Exception as exc:
                _log_once(f"[torch.compile] Could not enable suppress_errors: {exc}")

        compile_kwargs = {
            "backend": backend,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
        }
        if mode:
            compile_kwargs["mode"] = mode
        if options:
            compile_kwargs["options"] = options

        _log_once(
            "[torch.compile] Preparing "
            f"{name}: backend={backend}, mode={mode}, "
            f"fullgraph={fullgraph}, dynamic={dynamic}"
        )
        self.compiled_fn = torch.compile(fn, **compile_kwargs)

    def __call__(self, *args, **kwargs):
        if not self.enabled:
            return self.fn(*args, **kwargs)

        try:
            return self.compiled_fn(*args, **kwargs)
        except Exception as exc:
            self.enabled = False
            self.failed = True
            self.failure_reason = repr(exc)
            _log_once(
                f"[torch.compile][warn] {self.name} failed; "
                f"falling back to eager. reason={exc}"
            )
            return self.fn(*args, **kwargs)


def configure_module_call_torch_compile(
    module,
    *,
    name: str,
    backend: str = "inductor",
    mode: str | None = "max-autotune-no-cudagraphs",
    fullgraph: bool = False,
    dynamic: bool | None = False,
    options: dict | None = None,
    suppress_errors: bool = True,
):
    if not torch.cuda.is_available():
        _log_once(f"[torch.compile] Skipping {name}: CUDA is not available")
        return None

    try:
        return SafeCompiledCallable(
            module,
            name=name,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            options=options,
            suppress_errors=suppress_errors,
        )
    except Exception as exc:
        _log_once(
            f"[torch.compile][warn] Could not prepare {name}; "
            f"continuing in eager mode. reason={exc}"
        )
        return None
