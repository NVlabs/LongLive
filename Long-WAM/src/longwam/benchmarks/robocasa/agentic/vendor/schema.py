# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/vendor/schema.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.


def _obj(properties):
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def response_schema(request_id=None):
    number = {"type": "number"}
    vector3 = {"type": "array", "items": number, "minItems": 3, "maxItems": 3}
    vector4 = {"type": "array", "items": number, "minItems": 4, "maxItems": 4}
    string = {"type": "string"}
    progress = _obj(
        dict(
            verified_completed={"type": "array", "items": string},
            currently_attempting=string,
            remaining={"type": "array", "items": string},
        )
    )
    assessment = _obj(
        dict(
            task_progress=progress,
            current_subgoal=string,
            execution_status={
                "type": "string",
                "enum": ["not_started", "progressing", "failed", "uncertain", "recovered"],
            },
            execution_evidence=string,
            expected_next_intent=string,
            predicted_next_intent=string,
            intent_status={"type": "string", "enum": ["aligned", "misaligned", "uncertain"]},
            intent_evidence=string,
        )
    )
    identifier = string if request_id is None else {"type": "string", "enum": [request_id]}
    schema = _obj(
        dict(
            request_id=identifier,
            mode={"type": "string", "enum": ["student", "edit", "eef", "stop"]},
            steps={"type": "integer", "minimum": 1, "maximum": 15},
            reason=string,
            edit=_obj(
                dict(
                    delta_position=vector3,
                    delta_rotation_vector=vector3,
                    gripper={"type": "string", "enum": ["keep", "open", "closed"]},
                )
            ),
            target=_obj(
                dict(position=vector3, quaternion_wxyz=vector4, gripper_closed={"type": "boolean"})
            ),
            assessment=assessment,
        )
    )
    for field in ("edit", "target"):
        arm = schema["properties"][field]
        schema["properties"][field] = _obj(dict(left=arm, right=arm))
    return schema


def direct_response_schema():
    target = response_schema()["properties"]["target"]
    return _obj(
        dict(
            request_id={"type": "string"},
            mode={"type": "string", "enum": ["eef"]},
            steps={"type": "integer", "minimum": 1, "maximum": 5},
            reason={"type": "string"},
            target=target,
        )
    )
