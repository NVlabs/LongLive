# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/contract.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""Fixed contract of the Agentic + Long-WAM hybrid rollout on RoboCasa-365 (PandaOmron).

Adapted from the RoboDojo/RoboLab hybrid framework (eval-of-gpt-6-astra-as-policy, MIT) to the
RoboCasa-365 control interface used by our Long-WAM evaluation (experiments/robocasa):

* control at 20 Hz; the native task horizon (RoboCasa dataset registry) is the step limit
* student (Long-WAM) proposal: 32 x 12 dataset-format actions
      [base(4), mode(1), eef_pos(3), eef_rot(3), gripper(1)], each in [-1, 1]
  executed through robosuite's BASIC composite controller: arm OSC_POSE deltas in the robot
  BASE frame scaled by +-0.05 m / +-0.5 rad per step, gripper +1 = close / -1 = open,
  mode > 0 = base mode (arm goal frozen), mode < 0 = arm mode
* teacher (GPT) corrections are bounded absolute EEF targets or offsets, converted to OSC deltas
  step by step from the measured pose (no IK needed: the controller is already Cartesian)
"""

from __future__ import annotations

CONTROL_HZ = 20
CONTROL_DT = 1.0 / CONTROL_HZ
ACTION_DIM = 12
STUDENT_HORIZON = 32  # Long-WAM chunk
STUDENT_MAX_STEPS = 15  # framework rule: execute 1..15 proposal steps per decision
CORRECTION_MAX_STEPS = 5  # framework rule: 1..5 corrected steps per decision
STUDENT_ONLY_REPLAN = 32  # our formal Long-WAM protocol (replan_steps=32)
MAX_CONSECUTIVE_CORRECTIONS = 3  # RoboCasa variant: then the student must get the arm back

# Framework safeguards (per decision) and per-step conversion limits.
MAX_TARGET_DISTANCE_M = 0.05
MAX_TARGET_ROTATION_RAD = 0.35
MAX_STEP_TRANSLATION_M = 0.02
MAX_STEP_ROTATION_RAD = 0.10
OSC_POS_SCALE_M = 0.05  # |action|=1 -> 5 cm commanded delta per control step
OSC_ROT_SCALE_RAD = 0.5  # |action|=1 -> 0.5 rad commanded delta per control step

# Dataset action layout.
A_BASE = slice(0, 4)
A_MODE = 4
A_POS = slice(5, 8)
A_ROT = slice(8, 11)
A_GRIP = 11

# 16-D dataset state layout (see longwam.benchmarks.robocasa.contract.STATE_FIELDS).
S_BASE_POS = slice(0, 3)
S_BASE_QUAT = slice(3, 7)  # xyzw (robosuite convention)
S_EEF_POS = slice(7, 10)  # relative to the robot base frame
S_EEF_QUAT = slice(10, 14)  # xyzw, relative to the robot base frame
S_GRIPPER = slice(14, 16)  # finger joint positions

CAMERA_ORDER = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
FRAME = "robot_base"
EEF_LINK = "gripper0_right_grip_site"

METHODS = ("hybrid_decompose",)
