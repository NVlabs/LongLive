# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM implementation.
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# End Long-WAM attribution.

"""CPU-only scientific checks used by every public benchmark launcher."""


def check_training_contract(cfg, *, world_size):
    keep = cfg.get("checkpoint_keep_last")
    if keep is not None and (isinstance(keep, bool) or not isinstance(keep, int) or keep < 1):
        raise ValueError("checkpoint_keep_last must be null or a positive integer")
    batch = int(cfg.batch_size)
    accumulation = int(cfg.gradient_accumulation_steps)
    if min(batch, accumulation, world_size) <= 0:
        raise ValueError("batch_size, accumulation and world_size must be positive")
    actual = batch * accumulation * world_size
    expected = cfg.get("expected_global_batch_size")
    if expected is not None and actual != int(expected):
        raise ValueError(
            f"Global batch mismatch: {world_size} ranks * {batch} samples * "
            f"{accumulation} accumulation = {actual}; recipe requires {expected}. "
            "Change batch_size/gradient_accumulation_steps to preserve the recipe, "
            "or explicitly set expected_global_batch_size=null for a custom experiment."
        )
    train = cfg.data.train
    domains = train.get("domains")
    nodes = list(domains.values()) if domains else [train]
    history = int(cfg.model.get("longwam_past_obs_size", 0))
    imagine = int(cfg.model.get("longwam_num_imagine_frames", 0))
    if cfg.model.get("longwam_joint_denoise", False):
        if imagine <= 0 or float(cfg.model.longwam_imagine_sigma) != 0.0:
            raise ValueError("CodeDenoise requires future latents and longwam_imagine_sigma=0")
    for node in nodes:
        past = int(node.get("past_obs_size", 0))
        if past != history:
            raise ValueError(f"Data/model history mismatch: {past} != {history}")
        if past % 16:
            raise ValueError("History must cover whole stride-4/VAE-4 latent blocks")
        if imagine and int(node.num_frames) != past + imagine * 16 + 1:
            raise ValueError("num_frames must equal history + imagined raw frames + current")
        if int(node.processor.action_output_dim) != int(cfg.model.action_dit_config.action_dim):
            raise ValueError("Action processor/model dimensions disagree")
        if int(node.processor.proprio_output_dim) != int(cfg.model.proprio_dim):
            raise ValueError("Proprioception processor/model dimensions disagree")
        if int(node.action_chunk) <= 0:
            raise ValueError("action_chunk must be positive")
        if "robocasa_gr1" in str(node.get("_target_", "")) and node.source_recipe != "teleop":
            raise ValueError(
                "Public GR1 training uses the Teleop release, not the old low-quality batch"
            )
    return {
        "world_size": world_size,
        "per_rank_batch": batch,
        "accumulation": accumulation,
        "global_batch": actual,
        "history": history,
        "imagination": imagine,
        "action_chunk": int(nodes[0].action_chunk),
        "learning_rate": float(cfg.learning_rate),
        "epochs": int(cfg.num_epochs),
        "max_steps": cfg.max_steps,
    }
