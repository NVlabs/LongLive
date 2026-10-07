# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/vendor/network_recovery.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""Same-thread wakeup eligibility. No simulator restart or action replay."""

import json

NETWORK_CONTINUE_DELAYS = (10,) * 20


def is_network_error(error):
    text = json.dumps(error, ensure_ascii=False).lower()
    # Account/config/safety failures must not become automatic model calls.
    if any(
        s in text
        for s in (
            "usage limit",
            "quota",
            "unauthorized",
            "authentication",
            "invalid api key",
            "permission denied",
            "safety",
            "five rejected",
            "exceeds 5 cm",
        )
    ):
        return False
    return any(
        s in text
        for s in (
            "stream disconnected",
            "responsestreamdisconnected",
            "error decoding response body",
            "network error",
            "connection reset",
            "connection closed",
            "connection timed out",
            "tls close_notify",
        )
    )


def closed_network_turn(response, thread_id, turn_id):
    """Timeout probe is read-only. Never wake an active or interrupted turn."""
    thread = (response or {}).get("thread", {})
    if thread.get("id") != thread_id or thread.get("status", {}).get("type") != "idle":
        return None
    turns = thread.get("turns", [])
    if not turns or turns[-1].get("id") != turn_id:
        return None
    turn = turns[-1]
    return turn if turn.get("status") == "failed" and is_network_error(turn.get("error")) else None
