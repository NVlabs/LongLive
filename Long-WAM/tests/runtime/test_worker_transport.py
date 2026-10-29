# SPDX-FileCopyrightText: Copyright (c) 2026 AuricTW
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM transport regression tests.

"""Exercise transport failures with real spawned workers and bounded test gates."""

import multiprocessing as mp
import threading

import pytest

from longwam.runtime.worker import RemoteInferenceError, SpawnedInferenceProcess


def initialize_echo(gates):
    """Return a CPU-only handler, optionally gated inside the child process."""

    def handle(operation, payload):
        if operation == "block":
            gates[0].set()
            if not gates[1].wait(10):
                raise RuntimeError("test gate was not released")
        elif operation == "fail":
            raise ValueError("test handler failure")
        return payload

    return handle, {}


def initialize_failure(_):
    """Fail before the worker has announced readiness."""
    raise ValueError("test startup failure")


class Unpicklable:
    """Reach the sender unchanged, then fail during actual pipe serialization."""

    def __init__(self, error_type=TypeError):
        """Select an ordinary serialization exception or a thread-exit signal."""
        self.error_type = error_type

    def __deepcopy__(self, memo):
        """Keep notify's ownership copy from triggering the serialization failure."""
        return self

    def __reduce__(self):
        """Expose the same failure mode as an unpicklable user request."""
        raise self.error_type("test serialization failure")


class GatedConnection:
    """Pause a send while leaving real pipe reads, writes and cleanup intact."""

    def __init__(self, connection):
        self.connection = connection
        self.entered = threading.Event()
        self.release = threading.Event()
        self.drained = threading.Event()
        self.waiting = threading.Event()

    def send(self, message):
        """Gate the first notification and mark when the backlog was forwarded."""
        if isinstance(message.get("payload"), Unpicklable) or message.get("payload") == "gate":
            self.entered.set()
            if not self.release.wait(10):
                raise RuntimeError("test sender gate was not released")
        self.connection.send(message)
        if message.get("payload") == "last":
            self.drained.set()

    def poll(self, timeout=0):
        """Observe entry into a real blocking receive-side wait."""
        if timeout > 0:
            self.waiting.set()
        return self.connection.poll(timeout)

    def __getattr__(self, name):
        """Delegate all operations other than send to the original connection."""
        return getattr(self.connection, name)


class BrokenSendConnection:
    """Inject a failed write without also closing the readable pipe endpoint."""

    def __init__(self, connection):
        self.connection = connection

    def send(self, message):
        """Fail the background send with an ordinary transport error."""
        raise BrokenPipeError("test broken send")

    def __getattr__(self, name):
        """Keep the real receive and cleanup paths for this failure case."""
        return getattr(self.connection, name)


class BrokenReceiveConnection:
    """Inject a read failure after the real connection has become readable."""

    def __init__(self, connection):
        self.connection = connection

    def recv(self):
        """Represent a connection that failed between readiness and receive."""
        raise BrokenPipeError("test broken receive")

    def __getattr__(self, name):
        """Leave readiness, sends and resource ownership on the real connection."""
        return getattr(self.connection, name)


@pytest.fixture
def worker():
    """Always reap the actual child process, including after an assertion fails."""
    client = SpawnedInferenceProcess(
        initialize_echo, None, name="transport-test", startup_timeout_s=20
    )
    try:
        yield client
    finally:
        client.close(force=True)


@pytest.mark.parametrize("failure", ["serialization", "broken_pipe", "system_exit"])
def test_sender_failure_completes_future_and_is_cached(worker, failure):
    """A terminal sender failure is ready, actionable and stable across reads."""
    if failure == "broken_pipe":
        worker._connection = BrokenSendConnection(worker._connection)
        payload = None
    else:
        payload = Unpicklable(SystemExit if failure == "system_exit" else TypeError)

    pending = worker.submit("echo", payload)
    worker._sender.join(timeout=5)
    assert not worker._sender.is_alive()
    assert pending.done()
    with pytest.raises(RemoteInferenceError, match="transport failed") as first:
        pending.result(timeout=1)
    with pytest.raises(RemoteInferenceError) as second:
        pending.result(timeout=0)
    assert second.value is first.value
    with pytest.raises(RemoteInferenceError, match="transport failed"):
        worker.submit("echo", "retry")
    assert worker._pending_request_id is None


def test_sender_failure_wakes_an_unbounded_result_wait(worker):
    """result() without a timeout must wake even while the remote pipe stays open."""
    connection = GatedConnection(worker._connection)
    worker._connection = connection
    finished = threading.Event()
    outcome = []
    pending = worker.submit("echo", Unpicklable())

    def receive():
        """Record the result on a thread so a regression cannot hang the suite."""
        try:
            pending.result()
        except BaseException as error:
            outcome.append(error)
        finally:
            finished.set()

    receiver = threading.Thread(target=receive, daemon=True)
    receiver.start()
    try:
        assert connection.entered.wait(5)
        assert connection.waiting.wait(5)
        assert not finished.is_set()
        connection.release.set()
        assert finished.wait(3), "sender failure left result() blocked"
        assert len(outcome) == 1
        assert isinstance(outcome[0], RemoteInferenceError)
        assert "transport failed" in str(outcome[0])
    finally:
        connection.release.set()
        worker.close(force=True)
        receiver.join(timeout=5)
    assert not receiver.is_alive()


def test_failed_notification_rejects_subsequent_request(worker):
    """Notification serialization failure must not enqueue a request to a dead sender."""
    worker.notify("echo", Unpicklable())
    worker._sender.join(timeout=5)
    with pytest.raises(RemoteInferenceError, match="transport failed"):
        worker.submit("echo", "retry")
    assert worker._pending_request_id is None


def test_failed_notification_wakes_the_inflight_request():
    """A sender failure must wake a pending request even when its handler is blocked."""
    context = mp.get_context("spawn")
    gates = context.Event(), context.Event()
    client = SpawnedInferenceProcess(
        initialize_echo, gates, name="notification-test", startup_timeout_s=20
    )
    try:
        pending = client.submit("block", "value")
        assert gates[0].wait(5)
        client.notify("echo", Unpicklable())
        client._sender.join(timeout=5)
        assert pending.done()
        with pytest.raises(RemoteInferenceError, match="transport failed"):
            pending.result(timeout=1)
    finally:
        gates[1].set()
        client.close(force=True)


def test_full_queue_rejects_request_without_poisoning_single_flight(worker):
    """Retry a rejected submit after the real bounded queue has drained."""
    connection = GatedConnection(worker._connection)
    worker._connection = connection
    try:
        worker.notify("echo", "gate")
        assert connection.entered.wait(5)
        for index in range(128):
            worker.notify("echo", "last" if index == 127 else index)
        with pytest.raises(RuntimeError, match="queue is full"):
            worker.submit("echo", "rejected")
        assert worker._pending_request_id is None
        connection.release.set()
        assert connection.drained.wait(5)
        assert worker.call("echo", "retry") == "retry"
    finally:
        connection.release.set()


def test_timeout_keeps_request_retryable():
    """A normal response timeout leaves the same future and request pending."""
    context = mp.get_context("spawn")
    gates = context.Event(), context.Event()
    with SpawnedInferenceProcess(
        initialize_echo, gates, name="timeout-test", startup_timeout_s=20
    ) as client:
        pending = client.submit("block", "result")
        try:
            assert gates[0].wait(5)
            with pytest.raises(TimeoutError):
                pending.result(timeout=0)
            assert client._pending_request_id is not None
            assert not pending.done()
        finally:
            gates[1].set()
        assert pending.result(timeout=5) == "result"
        assert pending.done()
        assert pending.result(timeout=0) == "result"
        assert client.call("echo", "next") == "next"


def test_handler_errors_keep_transport_usable(worker):
    """Application errors remain recoverable rather than becoming transport failures."""
    with pytest.raises(RemoteInferenceError, match="test handler failure"):
        worker.call("fail")
    assert worker.call("echo", "next") == "next"
    assert worker._sender.is_alive()


def test_remote_exit_completes_pending_future(worker):
    """An exited child wakes the receive side and produces a remote error."""
    worker._process.terminate()
    worker._process.join(timeout=5)
    pending = worker.submit("echo", None)
    assert pending.done()
    with pytest.raises(RemoteInferenceError):
        pending.result(timeout=3)


def test_read_failure_after_readiness_is_forwarded(worker):
    """A pipe closing after poll must still produce a cached transport error."""
    pending = worker.submit("echo", "response")
    assert worker._connection.poll(5)
    worker._connection = BrokenReceiveConnection(worker._connection)
    with pytest.raises(RemoteInferenceError, match="test broken receive") as error:
        pending.result(timeout=0)
    assert isinstance(error.value.__cause__, BrokenPipeError)
    assert worker._pending_request_id is None
    assert pending.done()


def test_close_after_sender_failure_is_quiet_and_reaps_resources(worker):
    """Normal cleanup must not replace a recorded sender failure with a new error."""
    worker.submit("echo", Unpicklable())
    worker._sender.join(timeout=5)
    worker.close()
    worker.close()
    assert not worker._sender.is_alive()
    assert not worker._process.is_alive()
    assert worker._connection.closed


def test_close_handles_sender_failure_during_shutdown(worker):
    """A failed shutdown write falls back to reaping instead of masking cleanup."""
    worker._connection = BrokenSendConnection(worker._connection)
    worker.close()
    assert not worker._process.is_alive()
    assert not worker._sender.is_alive()
    assert worker._connection.closed


def test_healthy_close_reaps_sender_and_child(worker):
    """Graceful shutdown still sends its acknowledgement and owns all resources."""
    assert worker.call("echo", 3) == 3
    worker.close()
    assert not worker._sender.is_alive()
    assert not worker._process.is_alive()
    assert worker._connection.closed
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit("echo")
    with pytest.raises(RuntimeError, match="closed"):
        worker.notify("echo")


def test_force_close_completes_pending_future():
    """Stopping an owned worker makes its pending future terminal and readable."""
    context = mp.get_context("spawn")
    gates = context.Event(), context.Event()
    client = SpawnedInferenceProcess(
        initialize_echo, gates, name="close-pending-test", startup_timeout_s=20
    )
    try:
        pending = client.submit("block", None)
        assert gates[0].wait(5)
        client.close(force=True)
        assert not client._process.is_alive()
        assert not client._sender.is_alive()
        assert pending.done()
        with pytest.raises(RemoteInferenceError):
            pending.result(timeout=0)
    finally:
        client.close(force=True)


def test_startup_error_is_preserved():
    """Cleanup before sender construction preserves the initializer exception."""
    with pytest.raises(RemoteInferenceError, match="test startup failure"):
        SpawnedInferenceProcess(initialize_failure, None, name="startup-test", startup_timeout_s=20)
