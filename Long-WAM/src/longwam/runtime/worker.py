# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Long-WAM contributor infrastructure, integrated and adapted from the source snapshot.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/runtime/worker.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# Licensed under the Apache License, Version 2.0 for NVIDIA changes; upstream terms are retained.

"""Single-flight spawned-process RPC for simulator/inference isolation.

The scheduler allows only one inference candidate in
flight.  A duplex pipe therefore gives them the small Future-like surface they
need without a receiver thread in the simulator process.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import copy
import queue
import threading
import traceback
from multiprocessing.connection import Connection
from typing import Any, Callable, Optional


class RemoteInferenceError(RuntimeError):
    """An inference worker failed during startup or request execution."""


def _worker_entry(
    connection: Connection,
    initializer: Callable[[Any], tuple[Callable[[str, Any], Any], dict[str, Any]]],
    init_payload: Any,
) -> None:
    try:
        handler, metadata = initializer(init_payload)
        connection.send(
            {
                "kind": "ready",
                "pid": os.getpid(),
                "metadata": dict(metadata),
            }
        )
        while True:
            message = connection.recv()
            if message["kind"] == "shutdown":
                connection.send({"kind": "stopped", "pid": os.getpid()})
                return
            if message["kind"] == "notify":
                handler(str(message["operation"]), message.get("payload"))
                continue
            if message["kind"] != "request":
                raise RuntimeError(f"unknown inference IPC message: {message['kind']!r}")
            request_id = int(message["request_id"])
            try:
                value = handler(str(message["operation"]), message.get("payload"))
            except BaseException as error:
                connection.send(
                    {
                        "kind": "error",
                        "request_id": request_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
            else:
                connection.send(
                    {
                        "kind": "result",
                        "request_id": request_id,
                        "value": value,
                    }
                )
    except EOFError:
        return
    except BaseException as error:
        try:
            connection.send(
                {
                    "kind": "startup_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        except BaseException:
            pass
    finally:
        connection.close()


class ProcessFuture:
    """Future-compatible handle for the client's only in-flight request."""

    def __init__(self, client: "SpawnedInferenceProcess", request_id: int) -> None:
        self._client = client
        self._request_id = int(request_id)
        self._resolved = False
        self._value: Any = None
        self._error: Optional[BaseException] = None

    def done(self) -> bool:
        return self._resolved or self._client._connection.poll(0.0)

    def result(self, timeout: Optional[float] = None) -> Any:
        if not self._resolved:
            try:
                self._value = self._client._receive(self._request_id, timeout)
            except TimeoutError:
                raise
            except BaseException as error:
                self._error = error
            self._resolved = True
        if self._error is not None:
            raise self._error
        return self._value


class SpawnedInferenceProcess:
    """One long-lived inference process started with CUDA-safe ``spawn``."""

    def __init__(
        self,
        initializer: Callable[[Any], tuple[Callable[[str, Any], Any], dict[str, Any]]],
        init_payload: Any,
        *,
        name: str,
        startup_timeout_s: float = 600.0,
    ) -> None:
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self._connection = parent
        self._process = context.Process(
            target=_worker_entry,
            args=(child, initializer, init_payload),
            name=name,
            daemon=False,
        )
        self._next_request_id = 1
        self._pending_request_id: Optional[int] = None
        self._closed = False
        self._process.start()
        child.close()
        if not parent.poll(float(startup_timeout_s)):
            self.close(force=True)
            raise TimeoutError(f"inference process {name!r} did not become ready")
        try:
            message = parent.recv()
        except EOFError as error:
            self.close(force=True)
            raise RemoteInferenceError(
                f"inference process {name!r} exited during startup with code "
                f"{self._process.exitcode}"
            ) from error
        if message.get("kind") != "ready":
            self.close(force=True)
            raise RemoteInferenceError(self._format_remote_error(message))
        self.pid = int(message["pid"])
        self.metadata = dict(message["metadata"])
        self._outbox = queue.Queue(maxsize=128)
        self._send_error = None
        self._sender = threading.Thread(target=self._send_loop, name=f"{name}-sender", daemon=True)
        self._sender.start()

    def _send_loop(self):
        try:
            while True:
                message = self._outbox.get()
                if message is None:
                    return
                self._connection.send(message)
        except (OSError, EOFError) as error:
            self._send_error = error

    def _enqueue(self, message):
        if self._send_error is not None:
            raise RemoteInferenceError(f"Inference transport failed: {self._send_error}")
        try:
            self._outbox.put_nowait(message)
        except queue.Full as error:
            raise RuntimeError(
                "Inference observation queue is full; reduce the control backlog"
            ) from error

    @staticmethod
    def _format_remote_error(message: dict[str, Any]) -> str:
        detail = f"{message.get('error_type', 'RemoteError')}: {message.get('error', '')}"
        remote_traceback = str(message.get("traceback", "")).strip()
        return (
            detail if not remote_traceback else f"{detail}\nRemote traceback:\n{remote_traceback}"
        )

    def submit(self, operation: str, payload: Any = None) -> ProcessFuture:
        if self._closed:
            raise RuntimeError("inference process is closed")
        if self._pending_request_id is not None:
            raise RuntimeError("only one inference request may be in flight")
        request_id = self._next_request_id
        self._next_request_id += 1
        self._pending_request_id = request_id
        self._enqueue(
            {
                "kind": "request",
                "request_id": request_id,
                "operation": str(operation),
                "payload": payload,
            }
        )
        return ProcessFuture(self, request_id)

    def call(self, operation: str, payload: Any = None) -> Any:
        return self.submit(operation, payload).result()

    def notify(self, operation: str, payload: Any = None) -> None:
        if self._closed:
            raise RuntimeError("inference process is closed")
        self._enqueue(
            {
                "kind": "notify",
                "operation": str(operation),
                "payload": copy.deepcopy(payload),
            }
        )

    def _receive(self, request_id: int, timeout: Optional[float]) -> Any:
        if self._pending_request_id != int(request_id):
            raise RuntimeError("inference response does not match the in-flight request")
        if timeout is not None and not self._connection.poll(float(timeout)):
            raise TimeoutError(f"inference request {request_id} timed out")
        try:
            message = self._connection.recv()
        except EOFError as error:
            raise RemoteInferenceError(
                f"inference process exited with code {self._process.exitcode}"
            ) from error
        finally:
            self._pending_request_id = None
        if message.get("kind") == "startup_error":
            raise RemoteInferenceError(self._format_remote_error(message))
        if int(message.get("request_id", -1)) != int(request_id):
            raise RemoteInferenceError("received an out-of-order inference response")
        if message.get("kind") == "error":
            raise RemoteInferenceError(self._format_remote_error(message))
        if message.get("kind") != "result":
            raise RemoteInferenceError(f"unexpected inference response: {message!r}")
        return message.get("value")

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if not force and self._pending_request_id is None and self._process.is_alive():
                self._enqueue({"kind": "shutdown"})
                if self._connection.poll(10.0):
                    self._connection.recv()
        finally:
            if force and self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=10.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=10.0)
            self._connection.close()
            if hasattr(self, "_sender"):
                try:
                    self._outbox.put_nowait(None)
                except queue.Full:
                    pass
                self._sender.join(timeout=1.0)

    def shutdown(self, wait: bool = True) -> None:
        # Executor-compatible spelling used by scheduler cleanup/tests.
        del wait
        self.close()

    def __enter__(self) -> "SpawnedInferenceProcess":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_obj: Any) -> None:
        self.close(force=exc is not None)


class _InlineInferenceWorker:
    """Private synchronous transport for diagnostics with an already loaded model.

    Reuses the same worker handler and scheduling engine; creates no second policy
    API and never runs asynchronous control.
    """
    def __init__(self, handler):
        self.handler = handler

    def call(self, operation, payload):
        return self.handler(operation, payload)

    def notify(self, operation, payload):
        self.call(operation, payload)

    def submit(self, operation, payload):
        from concurrent.futures import Future
        future = Future()
        try:
            future.set_result(self.call(operation, payload))
        except Exception as error:
            future.set_exception(error)
        return future

    def close(self, **kwargs):
        pass
