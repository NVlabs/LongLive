# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 Yu-Mool Shu and Lipxin Zheng
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: robocasa/FastWAM/experiments/robocasa_gpt6/session.py
# Source: eval-of-gpt-6-astra-as-policy @ 79f8be5905102d6b16000c0f02a9c2195b51bb61
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# Changes: RoboCasa integration snapshot; original embodied-policy MIT notice is retained.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

"""RoboCasa-365 episode session shared by every evaluation method.

Owns one RoboCasa gym environment (same construction, reset seeding and success criterion as
experiments/robocasa/eval_robocasa.py) and one AF_UNIX client to the unchanged Long-WAM policy
server. Every executed control step is forwarded to the policy server (``observe``) so the P4
history is identical to the formal Long-WAM evaluation, whichever process chose the action.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from longwam.benchmarks.robocasa import rollout as ev
from longwam.benchmarks.robocasa.contract import (
    ACTION_DIM,
    CAMERA_KEYS,
    dataset_action_to_canonical,
    extract_cameras,
    extract_dataset_state,
    sanitize_dataset_action,
)
from longwam.benchmarks.robocasa.agentic import contract as C
from longwam.benchmarks.robocasa.agentic.contract import CAMERA_ORDER


def quat_xyzw_to_wxyz(q):
    q = np.asarray(q, float)
    return np.array([q[3], q[0], q[1], q[2]])


def pose_from_state(state: np.ndarray) -> dict[str, Any]:
    """Measured EEF pose in the robot base frame, framework representation (wxyz)."""
    return dict(
        position=np.asarray(state[C.S_EEF_POS], float).tolist(),
        quaternion_wxyz=quat_xyzw_to_wxyz(state[C.S_EEF_QUAT]).tolist(),
        frame=C.FRAME,
        link=C.EEF_LINK,
        gripper_qpos=np.asarray(state[C.S_GRIPPER], float).tolist(),
    )


def _rot(q_wxyz):
    q = np.asarray(q_wxyz, float)
    return Rotation.from_quat(q[[1, 2, 3, 0]])


def robot_only_fk(state: np.ndarray, actions: np.ndarray) -> list[dict[str, Any]]:
    """Integrate the proposal's OSC deltas from the measured pose (base frame, base motion ignored).

    This is the commanded goal trajectory, not a physics simulation: contact, tracking error and
    base motion are not modelled. Steps in base mode leave the arm goal unchanged.
    """
    pos = np.asarray(state[C.S_EEF_POS], float).copy()
    rot = _rot(quat_xyzw_to_wxyz(state[C.S_EEF_QUAT]))
    out = []
    for i, a in enumerate(np.asarray(actions, np.float32)):
        arm_mode = a[C.A_MODE] < 0
        base_moving = bool(np.any(np.abs(a[C.A_BASE]) > 1e-6)) and not arm_mode
        if arm_mode:
            pos = pos + a[C.A_POS] * C.OSC_POS_SCALE_M
            rot = Rotation.from_rotvec(a[C.A_ROT] * C.OSC_ROT_SCALE_RAD) * rot
        q = rot.as_quat()  # xyzw
        out.append(
            dict(
                index=i,
                position=pos.tolist(),
                quaternion_wxyz=[q[3], q[0], q[1], q[2]],
                gripper_closed=bool(a[C.A_GRIP] > 0),
                arm_mode=bool(arm_mode),
                base_motion=bool(base_moving),
            )
        )
    return out


def eef_target_to_action(
    state: np.ndarray, target: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """One bounded OSC step toward an absolute base-frame EEF target (recomputed per ACK)."""
    cur_p = np.asarray(state[C.S_EEF_POS], float)
    cur_r = _rot(quat_xyzw_to_wxyz(state[C.S_EEF_QUAT]))
    dp = np.asarray(target["position"], float) - cur_p
    dist = float(np.linalg.norm(dp))
    if dist > C.MAX_STEP_TRANSLATION_M:
        dp = dp * (C.MAX_STEP_TRANSLATION_M / dist)
    dr = (_rot(target["quaternion_wxyz"]) * cur_r.inv()).as_rotvec()
    ang = float(np.linalg.norm(dr))
    if ang > C.MAX_STEP_ROTATION_RAD:
        dr = dr * (C.MAX_STEP_ROTATION_RAD / ang)
    action = np.zeros(ACTION_DIM, np.float32)
    action[C.A_MODE] = -1.0  # arm mode
    action[C.A_POS] = np.clip(dp / C.OSC_POS_SCALE_M, -1.0, 1.0)
    action[C.A_ROT] = np.clip(dr / C.OSC_ROT_SCALE_RAD, -1.0, 1.0)
    action[C.A_GRIP] = 1.0 if target["gripper_closed"] else -1.0
    diagnostics = dict(
        remaining_distance_m=dist,
        remaining_rotation_rad=ang,
        step_translation_m=float(np.linalg.norm(dp)),
        step_rotation_rad=float(np.linalg.norm(dr)),
    )
    return action, diagnostics


class RoboCasaSession:
    """One task-bound environment plus the Long-WAM policy connection."""

    def __init__(
        self,
        *,
        task: str,
        split: str,
        env_seed: int,
        policy_socket: Path,
        use_policy: bool = True,
        response_timeout_seconds: float = 900.0,
        startup_timeout_seconds: float = 900.0,
    ):
        registry, get_task_horizon, gym_make, convert_action = ev._load_robocasa_runtime()
        self.task, self.split, self.env_seed = task, split, int(env_seed)
        self.group = ev.task_group(registry, task)
        self.horizon = int(get_task_horizon(task))
        self.convert_action = convert_action
        self.env = gym_make(f"robocasa/{task}", split=split, seed=self.env_seed)
        # Policy-server contract: exactly ONE observe/infer per control step, strictly sequential.
        # infer() covers the step it is called at; run_* forward the intermediate post-step
        # observations and leave the last one for the next infer().
        self.use_policy = bool(use_policy)
        self.policy = None
        if self.use_policy:
            self.policy = ev.UnixPolicyClient(
                policy_socket,
                response_timeout_seconds=response_timeout_seconds,
                startup_timeout_seconds=startup_timeout_seconds,
            )
            self.policy.connect()
            self.policy.ping()
        self.episode_id = None
        self.step_id = 0
        self.observation = None
        self.render_resets: list[
            int
        ] = []  # control steps at which the EGL offscreen context was rebuilt
        self._dead_render_contexts: list[Any] = []
        self.instruction = ""
        self.success = False
        self.done = False
        self.inference_calls = 0
        self.frames: list[np.ndarray] = []
        self.executed: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ episode lifecycle
    def reset(self, episode_index: int) -> dict[str, Any]:
        self.episode_index = int(episode_index)
        self.episode_id = f"{self.task}:{self.env_seed}:{self.episode_index}"
        self.reset_seed = ev.derive_episode_reset_seed(
            self.env_seed, self.split, self.task, self.episode_index
        )
        random.seed(self.reset_seed)
        np.random.seed(self.reset_seed & 0xFFFFFFFF)
        self.observation, _ = self.env.reset(seed=self.reset_seed)
        self._heal_renderer_if_corrupted()
        self.instruction = str(self.observation.get(ev.TASK_DESCRIPTION_KEY, "")).strip()
        if not self.instruction:
            raise ValueError("RoboCasa reset produced an empty task instruction.")
        if self.use_policy:
            self.policy.reset(self.episode_id)
        self.step_id, self.success, self.done, self.inference_calls = 0, False, False, 0
        self.frames, self.executed = [self._frame()], []
        self.started = time.monotonic()
        return dict(
            episode_id=self.episode_id,
            reset_seed=self.reset_seed,
            instruction=self.instruction,
            horizon=self.horizon,
            group=self.group,
        )

    # ------------------------------------------------------------------ renderer self-healing
    NOISE_GRADIENT_THRESHOLD = (
        40.0  # mean |adjacent pixel difference|: real renders ~5 (p50), driver noise ~75
    )

    @staticmethod
    def cameras_corrupted(cams: dict[str, np.ndarray]) -> bool:
        """Broken EGL renderers (GPU starvation / driver fault) return either random noise on every camera
        or one frozen image repeated on all cameras."""
        v = [np.asarray(cams[k], np.int16) for k in CAMERA_KEYS]
        if len(v) != 3:
            return False
        m = [float(a.mean()) for a in v]
        sd = [float(a.std()) for a in v]
        frozen = (
            max(m) - min(m) < 0.5 and max(sd) - min(sd) < 0.5
        )  # three real viewpoints never match this closely
        noise = any(
            (np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()) / 2
            > RoboCasaSession.NOISE_GRADIENT_THRESHOLD
            for a in v
        )
        return bool(frozen or noise)

    def rebuild_renderer(self) -> None:
        """Recreate the robosuite offscreen render context and re-render the current observation."""
        from robosuite.utils.binding_utils import MjRenderContextOffscreen

        wrapper = self.env.unwrapped  # robocasa gym wrapper; .env is the robosuite environment
        rs_env = wrapper.env
        sim = rs_env.sim
        old = sim._render_context_offscreen
        if old is not None:
            # Free the old context explicitly BEFORE creating the new one: robosuite's destructor calls
            # eglReleaseThread(), which would also unbind a freshly created context and make MjrContext
            # creation fail with "Default framebuffer is not complete".
            sim._render_context_offscreen = None
            try:
                old.con.free()
            except Exception:
                pass
            try:
                old.gl_ctx.free()
            except Exception:
                pass

            class _Dead(
                type(old)
            ):  # keep the object alive without a second (double-free) destructor
                def __del__(self):
                    pass

            old.__class__ = _Dead
            self._dead_render_contexts.append(old)
        context = MjRenderContextOffscreen(
            sim, device_id=rs_env.render_gpu_device_id
        )  # registers itself on sim
        if sim._render_context_offscreen is not context:
            sim.add_render_context(context)
        sim._render_context_offscreen.vopt.geomgroup[0] = 1 if rs_env.render_collision_mesh else 0
        sim._render_context_offscreen.vopt.geomgroup[1] = 1 if rs_env.render_visual_mesh else 0
        self.observation = wrapper.get_observation(rs_env._get_observations(force_update=True))

    def _heal_renderer_if_corrupted(self) -> None:
        if not self.cameras_corrupted(extract_cameras(self.observation)):
            return
        self.render_resets.append(self.step_id)
        for attempt in range(3):
            self.rebuild_renderer()
            if not self.cameras_corrupted(extract_cameras(self.observation)):
                print(
                    json.dumps(
                        dict(
                            event="render_context_rebuilt",
                            step_id=self.step_id,
                            attempt=attempt + 1,
                        )
                    ),
                    flush=True,
                )
                return
        print(
            json.dumps(dict(event="render_context_still_corrupted", step_id=self.step_id)),
            flush=True,
        )

    def teacher_views(self, size: int) -> dict[str, np.ndarray]:
        """Extra high-resolution renders of the three cameras for the GPT teacher only (policy input unchanged)."""
        sim = self.env.unwrapped.env.sim
        out = {}
        for camera in CAMERA_ORDER:
            img = sim.render(width=size, height=size, camera_name=camera)
            out[camera] = np.ascontiguousarray(img[::-1])
        return out

    def _frame(self) -> np.ndarray:
        cams = extract_cameras(self.observation)
        return np.concatenate([cams[k] for k in CAMERA_KEYS], axis=1)  # [256, 768, 3]

    # ------------------------------------------------------------------ observation packet
    def state(self) -> np.ndarray:
        return extract_dataset_state(self.observation)

    def cameras(self) -> dict[str, np.ndarray]:
        return {k.removeprefix("video."): v for k, v in extract_cameras(self.observation).items()}

    def remaining_steps(self) -> int:
        return max(0, self.horizon - self.step_id)

    # ------------------------------------------------------------------ student inference
    def infer(self, prompt: str | None = None) -> np.ndarray:
        """Fresh Long-WAM inference from the current observation (records it in the history)."""
        if self.done:
            raise RuntimeError("Episode finished.")
        prompt = self.instruction if prompt is None else prompt
        result = self.policy.infer(
            episode_id=self.episode_id,
            control_step=self.step_id,
            observation=self.observation,
            prompt=prompt,
        )
        self.inference_calls += 1
        actions = np.asarray(result.actions, np.float32)
        if actions.shape != (C.STUDENT_HORIZON, ACTION_DIM):
            raise ValueError(
                f"Long-WAM returned {actions.shape}, expected {(C.STUDENT_HORIZON, ACTION_DIM)}"
            )
        return actions

    def observe_only(self) -> None:
        """Forward the current observation to the policy history without inferring."""
        self.policy.observe(
            episode_id=self.episode_id, control_step=self.step_id, observation=self.observation
        )

    # ------------------------------------------------------------------ execution
    def step(
        self, dataset_action: np.ndarray, *, source: str, meta: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute ONE dataset-format action. The observation used for this step has already been
        sent to the policy (by infer() or observe_only()); the caller decides how the post-step
        observation is forwarded (observe_only for intermediate steps, the next infer otherwise)."""
        if self.done:
            raise RuntimeError("Episode finished.")
        sanitized = sanitize_dataset_action(np.asarray(dataset_action, np.float32))
        canonical = dataset_action_to_canonical(sanitized)
        state_before = self.state()
        self.observation, _, _, _, info = self.env.step(self.convert_action(canonical))
        self.step_id += 1
        self._heal_renderer_if_corrupted()
        self.frames.append(self._frame())
        self.success = bool(ev.success_from_info(info))
        truncated = (not self.success) and self.step_id >= self.horizon
        self.done = self.success or truncated
        row = dict(
            step_id=self.step_id,
            source=source,
            executed_action=sanitized.tolist(),
            state_before=state_before.tolist(),
            state_after=self.state().tolist(),
            success=self.success,
            terminated=self.success,
            truncated=truncated,
            **(meta or {}),
        )
        self.executed.append(row)
        return row

    def _forward_intermediate(self, more_steps_follow: bool) -> None:
        if self.use_policy and more_steps_follow and not self.done:
            self.observe_only()

    def run_student_prefix(
        self, actions: np.ndarray, steps: int, *, source: str = "student"
    ) -> list[dict[str, Any]]:
        rows = []
        n = int(steps)
        for i in range(n):
            rows.append(self.step(actions[i], source=source, meta=dict(proposal_index=i)))
            if self.done:
                break
            self._forward_intermediate(i + 1 < n)
        return rows

    def run_eef_targets(
        self, targets: list[dict[str, Any]], *, source: str
    ) -> list[dict[str, Any]]:
        rows = []
        for i, target in enumerate(targets):
            action, diag = eef_target_to_action(self.state(), target)
            rows.append(
                self.step(
                    action, source=source, meta=dict(edited_target=target, eef_diagnostics=diag)
                )
            )
            if self.done:
                break
            self._forward_intermediate(i + 1 < len(targets))
        return rows

    # ------------------------------------------------------------------ results
    def summary(self) -> dict[str, Any]:
        return dict(
            episode_id=self.episode_id,
            task=self.task,
            group=self.group,
            split=self.split,
            env_seed=self.env_seed,
            episode_index=self.episode_index,
            reset_seed=self.reset_seed,
            instruction=self.instruction,
            horizon=self.horizon,
            steps=self.step_id,
            success=self.success,
            complete=self.done,
            inference_calls=self.inference_calls,
            render_resets=list(self.render_resets),
            duration_seconds=time.monotonic() - self.started,
            source_steps={
                s: sum(1 for r in self.executed if r["source"] == s)
                for s in sorted({r["source"] for r in self.executed})
            },
        )

    def write_video(self, path: Path, fps: int = C.CONTROL_HZ) -> None:
        import imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7) as w:
            for f in self.frames:
                w.append_data(f)

    def close(self) -> None:
        try:
            if self.policy is not None:
                self.policy.close()
        finally:
            self.env.close()
