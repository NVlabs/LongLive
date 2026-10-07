# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/vendor/io.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# Changes: Redact credential fields and token formats in persisted audit copies.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""Small file helpers shared by the three independent components."""

import hashlib
import json
import os
import re
from pathlib import Path
import uuid


_SENSITIVE_FIELD = re.compile(
    r"(?i)^(?:(?:[a-z0-9]+[_-])*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"id[_-]?token|auth[_-]?token|secret[_-]?key)|authorization|password|client[_-]?secret)$"
)
_ASSIGNMENT = re.compile(
    r"(?i)\b((?:[a-z0-9]+[_-])*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"id[_-]?token|auth[_-]?token|secret[_-]?key)|authorization|password|client[_-]?secret)"
    r"([\"']?\s*[:=]\s*)(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|(?:Bearer\s+)?[^\s,;}]+)"
)
_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|"
    r"nvapi-[A-Za-z0-9_-]{20,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"
)


def redact_credentials(value):
    """Redact known credential fields/formats from audit copies, never wire data."""
    if isinstance(value, dict):
        return {key: "[REDACTED]" if _SENSITIVE_FIELD.fullmatch(str(key))
                else redact_credentials(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_credentials(item) for item in value]
    if isinstance(value, str):
        value = _TOKEN.sub("[REDACTED]", value)
        return _ASSIGNMENT.sub(lambda match: match[1] + match[2] + "[REDACTED]", value)
    return value


def redact_log_line(line):
    try:
        return json.dumps(redact_credentials(json.loads(line)), ensure_ascii=False) + "\n"
    except (ValueError, TypeError):
        return redact_credentials(line)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    # Different containers can publish the same derived ledger. A fixed .tmp
    # name is unsafe even when callers use advisory locks on shared storage.
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    stream = temporary.open("x")  # Preserve the normal open/umask permissions.
    try:
        with stream:
            json.dump(redact_credentials(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        # Only our uniquely created staging file; never another writer's file.
        temporary.unlink(missing_ok=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class InputError(ValueError):
    """Rejected before inference/physics: safe for the caller to correct arguments."""
