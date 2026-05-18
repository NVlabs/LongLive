# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple
import random
import torch

from model.base import BaseModel
from pipeline import CausalDiffusionInferencePipeline
from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalDiffusion(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the Diffusion loss module.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.teacher_forcing = getattr(args, "teacher_forcing", False)
        # Noise augmentation in teacher forcing, we add small noise to clean context latents
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)

        self.args = args
        self.device = device
        self.inference_pipeline = None

        # Error recycling (SVI-style error buffer)
        # When ``enable_position_bucketing`` is true, each rank holds a 2D
        # buffer ``(local_block_position × timestep)``. The pos dimension only
        # covers the LOCAL slice of the sequence this rank is responsible for
        # (no cross-SP-rank pos sharing — those positions are simply not
        # reachable by this rank during forward), so memory cost scales as
        # ``num_blocks_global / sp_size`` instead of ``num_blocks_global``.
        # ``global_block_offset`` is recorded for logging only.
        # During the first ``buffer_warmup_iter`` global steps, errors are
        # all-gathered across the DP group (ranks with the same SP rank but
        # different DP replicas), so each rank's local pos buckets fill up
        # ``dp_size`` × faster without any wasted bandwidth.
        self.error_buffer = None
        self.noise_error_buffer = None
        self.er_num_blocks = 0       # local; >0 means 2D position-bucketed
        self.er_block_offset = 0     # global block offset for THIS rank
        er_cfg = getattr(args, "error_recycling", None)
        if er_cfg is not None and getattr(er_cfg, "enabled", False):
            from utils.error_buffer import build_error_buffer
            cfg_dict = er_cfg if isinstance(er_cfg, dict) else dict(er_cfg)
            cfg_dict.setdefault("num_train_timesteps", self.num_train_timestep)
            sp_size = int(getattr(args, "sequence_parallel_size", 1) or 1)
            if cfg_dict.get("enable_position_bucketing", False):
                shape = list(getattr(args, "image_or_video_shape", [1, 0]))
                total_frames = int(shape[1]) if len(shape) > 1 else 0
                assert total_frames > 0 and self.num_frame_per_block > 0, (
                    "enable_position_bucketing=true requires "
                    "image_or_video_shape[1] and num_frame_per_block to be set."
                )
                num_blocks_global = total_frames // self.num_frame_per_block
                assert num_blocks_global % sp_size == 0, (
                    f"num_blocks_global ({num_blocks_global}) must be divisible "
                    f"by sequence_parallel_size ({sp_size})."
                )
                self.er_num_blocks = num_blocks_global // sp_size  # local
                # Determine this rank's SP index → global block offset.
                if sp_size > 1:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        sp_rank = dist.get_rank() % sp_size
                    else:
                        sp_rank = 0
                else:
                    sp_rank = 0
                self.er_block_offset = sp_rank * self.er_num_blocks

            # Shard timestep buckets across SP ranks: each SP rank only
            # stores t_bucket % sp_size == sp_rank, cutting per-rank CPU
            # memory by ~sp_size.  This uses the same SP dimension that
            # already splits positions in 2D mode, so both 1D and 2D
            # follow one save/load pattern (per sp_rank).
            import torch.distributed as dist
            if sp_size > 1 and dist.is_initialized():
                er_shard_rank = dist.get_rank() % sp_size
                er_shard_size = sp_size
            else:
                er_shard_rank = 0
                er_shard_size = 1

            self.error_buffer = build_error_buffer(
                cfg_dict, num_blocks=self.er_num_blocks,
                global_block_offset=self.er_block_offset,
                shard_rank=er_shard_rank, shard_size=er_shard_size,
            )
            self.noise_error_buffer = build_error_buffer(
                cfg_dict, num_blocks=self.er_num_blocks,
                global_block_offset=self.er_block_offset,
                shard_rank=er_shard_rank, shard_size=er_shard_size,
            )
            self.er_context_inject_prob = float(cfg_dict.get("context_inject_prob", 0.9))
            self.er_latent_inject_prob = float(cfg_dict.get("latent_inject_prob", 0.0))
            self.er_noise_inject_prob = float(cfg_dict.get("noise_inject_prob", 0.0))
            self.er_clean_prob = float(cfg_dict.get("clean_prob", 0.0))
            self.er_clean_buffer_update_prob = float(cfg_dict.get("clean_buffer_update_prob", 0.1))
            self.er_start_step = int(cfg_dict.get("start_step", 0))
            self.er_buffer_warmup_iter = int(cfg_dict.get("buffer_warmup_iter", 50))
            self.er_skip_block_0 = bool(cfg_dict.get("skip_block_0", False))

    def _initialize_models(self, args, device):
        model_name = getattr(args.model_kwargs, "model_name", "Wan2.2-TI2V-5B")
        if "5B" not in model_name:
            raise ValueError(f"Only Wan2.2-TI2V-5B is supported in this release, got {model_name}")
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        loss_mask_global_valid_count: torch.Tensor = None,
        global_step: int = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - loss_mask: optional tensor of shape [B, F] with 1.0 for valid frames and 0.0 for padded frames.
                         Under Sequence Parallel this is already the local chunk.
            - loss_mask_global_valid_count: optional scalar tensor with the total valid count across all SP ranks.
                         When provided (SP mode), used as the denominator instead of the local loss_mask.sum().
            - global_step: current training step, used for error recycling delayed start.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        batch_size, num_frame = image_or_video_shape[:2]

        noise = torch.randn_like(clean_latent)
        # Step 2: Randomly sample a timestep and add noise to denoiser inputs
        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=False
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)

        # Step 2.5 & 3.5: Error recycling — clean_prob acts as a master switch.
        # When clean_prob fires, skip ALL error injection and use pristine input;
        # otherwise each injection type rolls its own probability and is gated
        # only by whether the corresponding buffer has any samples (SVI behavior).
        #
        # NOTE on rank-sync: random.random() below is INTENTIONALLY independent
        # across ranks. None of these decisions guard a collective call (we use
        # SVI's pattern of unconditional all_gather + local random replay in
        # Step 5), so per-rank divergence here only affects which slice of data
        # gets corrupted on which rank — perfectly safe under DP+SP, and matches
        # SVI's behavior exactly.
        er_latent_injected = False
        er_noise_injected = False
        er_injected = False
        er_use_clean = False
        er_ready = (
            self.error_buffer is not None
            and (global_step is None or global_step >= self.er_start_step)
        )
        if er_ready and self.er_clean_prob > 0 and random.random() < self.er_clean_prob:
            er_use_clean = True

        # Noise error injection (SVI's noise_prob): corrupt noise input.
        # training_target is then computed with the corrupted noise so the
        # model learns to predict a self-correcting velocity (SVI Eq. logic).
        noise_for_train = noise
        if (
            er_ready and not er_use_clean
            and self.er_noise_inject_prob > 0
            and not self.noise_error_buffer.is_empty()
            and random.random() < self.er_noise_inject_prob
        ):
            noise_for_train = self._inject_noise_error_buffer(
                noise, index, batch_size, num_frame
            )
            er_noise_injected = True

        # Latent error injection (SVI's latent_prob): corrupt clean_latent
        # before noising. training_target keeps pointing to ORIGINAL clean_latent.
        clean_latent_for_noise = clean_latent
        if (
            er_ready and not er_use_clean
            and self.er_latent_inject_prob > 0
            and not self.error_buffer.is_empty()
            and random.random() < self.er_latent_inject_prob
        ):
            clean_latent_for_noise = self._inject_latent_error_buffer(
                clean_latent, index, batch_size, num_frame
            )
            er_latent_injected = True

        noisy_latents = self.scheduler.add_noise(
            clean_latent_for_noise.flatten(0, 1),
            noise_for_train.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.scheduler.training_target(clean_latent, noise_for_train, timestep)

        # Step 3: Noise augmentation, also add small noise to clean context latents
        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                0,
                self.noise_augmentation_max_timestep,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(dtype=self.dtype, device=self.device)
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = None

        # Step 3.5: Error recycling — inject sampled errors into clean prefix.
        # 2D mode: per-position (random timestep). 1D mode: SVI global sampling.
        if (
            er_ready and not er_use_clean
            and not self.error_buffer.is_empty()
            and random.random() < self.er_context_inject_prob
        ):
            clean_latent_aug = self._inject_error_buffer(
                clean_latent_aug, index, batch_size, num_frame
            )
            er_injected = True

        # Compute loss
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent_aug if self.teacher_forcing else None,
            aug_t=timestep_clean_aug if self.teacher_forcing else None,
        )
        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction='none'
        ).mean(dim=(2, 3, 4))
        loss = loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        if loss_mask is not None:
            loss = loss * loss_mask
            valid_count = loss_mask_global_valid_count if loss_mask_global_valid_count is not None else loss_mask.sum()
            loss = loss.sum() / valid_count.clamp(min=1.0)
        else:
            loss = loss.mean()

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach()
        }

        # Step 5: Store prediction errors into error buffer.
        #
        # SVI-style two-phase pattern (avoids the rank-divergent collective
        # deadlock that an ``if random.random() < p: all_gather()`` would
        # introduce):
        #   PHASE A (collective, UNCONDITIONAL): every rank reaches the
        #     all_gather call together so NCCL stays in sync.
        #   PHASE B (local, GATED): each rank independently decides whether
        #     to actually replay the gathered items into its buffer. The
        #     gate uses random.random() per rank — divergence here is fine
        #     because no further collective follows.
        if self.error_buffer is not None:
            with torch.no_grad():
                # latent error: x0_pred - clean_latent ≡ -σ(v_pred - v_gt) — used for context/latent injection
                latent_err = x0_pred.detach() - clean_latent.detach()
                # noise error: SVI definition is (1-σ)(v_pred - v_gt) so the buffer entry,
                # when later added directly to noise, equals ε_pred - ε_gt.
                sigma = self.scheduler.sigmas.to(flow_pred.device)[index].reshape(
                    batch_size, num_frame, 1, 1, 1
                ).to(flow_pred.dtype)
                noise_err = (flow_pred.detach() - training_target.detach()) * (1.0 - sigma)

                use_distributed = (
                    global_step is not None
                    and global_step <= self.er_buffer_warmup_iter
                )

                # === PHASE A: collective — runs on EVERY rank, no gating ===
                if use_distributed:
                    lat_items = self._gather_errors_for_buffer(
                        self.error_buffer, latent_err, index, batch_size, num_frame
                    )
                    noise_items = self._gather_errors_for_buffer(
                        self.noise_error_buffer, noise_err, index, batch_size, num_frame
                    )
                else:
                    lat_items = self._collect_local_items(
                        self.error_buffer, latent_err, index, batch_size, num_frame
                    )
                    noise_items = self._collect_local_items(
                        self.noise_error_buffer, noise_err, index, batch_size, num_frame
                    )

                # === PHASE B: local replay — random.random() per-rank is OK ===
                # When the input was clean (low-error), only update buffer with
                # small probability to avoid flooding it with near-zero samples
                # (SVI: clean_buffer_update_prob).
                should_update = True
                if er_use_clean and random.random() >= self.er_clean_buffer_update_prob:
                    should_update = False
                if should_update:
                    self._apply_gathered_items(self.error_buffer, lat_items)
                    self._apply_gathered_items(self.noise_error_buffer, noise_items)
            buf_stats = self.error_buffer.stats()
            noise_buf_stats = self.noise_error_buffer.stats()
            log_dict["er_total_added"] = buf_stats["total_added"]
            log_dict["er_filled_buckets"] = buf_stats["filled_buckets"]
            log_dict["er_total_entries"] = buf_stats["total_entries"]
            log_dict["er_noise_total_entries"] = noise_buf_stats["total_entries"]
            log_dict["er_injected"] = er_injected
            log_dict["er_latent_injected"] = er_latent_injected
            log_dict["er_noise_injected"] = er_noise_injected

        return loss, log_dict


    def _inject_error_buffer(self, clean_latent_aug, index, batch_size, num_frame):
        """Inject errors into the clean prefix (E_img).

        2D (position-bucketed): the i-th LOCAL prefix block draws from
        ``buckets[(i, *)]`` with a RANDOM timestep — the clean prefix is
        the product of full ODE integration so its accumulated error can
        come from any noise level, but its magnitude scales with the
        block's global position. Note ``skip_block_0`` is interpreted in
        the GLOBAL frame: only the very first SP rank may skip its block 0.

        1D (timestep-bucketed): falls back to SVI ``sample_global``.
        """
        block_size = self.num_frame_per_block
        num_blocks = num_frame // block_size
        result = clean_latent_aug.clone()
        for b in range(batch_size):
            for blk in range(num_blocks):
                if self.er_skip_block_0 and (self.er_block_offset + blk) == 0:
                    continue
                if self.er_num_blocks > 0:
                    err = self.error_buffer.sample_pos_any_t(
                        blk, device=result.device, dtype=result.dtype
                    )
                else:
                    err = self.error_buffer.sample_global(
                        device=result.device, dtype=result.dtype
                    )
                if err is not None:
                    start = blk * block_size
                    end = start + block_size
                    result[b, start:end] = result[b, start:end] + err
        return result

    def _inject_latent_error_buffer(self, clean_latent, index, batch_size, num_frame):
        """Inject errors into clean_latent before noising (E_vid).

        Matches BOTH block_position (LOCAL) and timestep when the buffer is
        2D, else only timestep (SVI default).
        """
        block_size = self.num_frame_per_block
        num_blocks = num_frame // block_size
        index_per_block = index[:, ::block_size]
        result = clean_latent.clone()
        for b in range(batch_size):
            for blk in range(num_blocks):
                t_idx = index_per_block[b, blk].item()
                pos = blk if self.er_num_blocks > 0 else None
                err = self.error_buffer.sample(
                    t_idx, device=result.device, dtype=result.dtype,
                    block_pos=pos,
                )
                if err is not None:
                    start = blk * block_size
                    end = start + block_size
                    result[b, start:end] = result[b, start:end] + err
        return result

    def _inject_noise_error_buffer(self, noise, index, batch_size, num_frame):
        """Inject errors into the noise (E_noise).

        Same matching strategy as ``_inject_latent_error_buffer`` but reads
        from the dedicated noise buffer.
        """
        block_size = self.num_frame_per_block
        num_blocks = num_frame // block_size
        index_per_block = index[:, ::block_size]
        result = noise.clone()
        for b in range(batch_size):
            for blk in range(num_blocks):
                t_idx = index_per_block[b, blk].item()
                pos = blk if self.er_num_blocks > 0 else None
                err = self.noise_error_buffer.sample(
                    t_idx, device=result.device, dtype=result.dtype,
                    block_pos=pos,
                )
                if err is not None:
                    start = blk * block_size
                    end = start + block_size
                    result[b, start:end] = result[b, start:end] + err
        return result

    def _gather_errors_for_buffer(
        self, buffer, error, index, batch_size, num_frame
    ):
        """All-gather errors/timesteps across the appropriate group and return
        a list of ready-to-add ``(err_block, t_idx, pos_or_None)`` items.

        ★ This is a COLLECTIVE — every rank MUST reach this call together.
        The caller is responsible for invoking it unconditionally during the
        warmup window (just like SVI's ``all_gather`` outside the random
        ``if`` blocks). Random decisions about whether to actually consume
        the returned items belong to ``_apply_gathered_items`` instead.

        Group selection mirrors SVI's intent:
          * **2D (num_blocks > 0)** — DP group only. Other SP ranks' samples
            map to positions unreachable by this rank, so cross-SP gather
            wastes bandwidth.
          * **1D (num_blocks == 0)** — WORLD group (SVI default). Buckets
            are pos-agnostic so every rank's errors are valid samples.
        """
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return self._collect_local_items(buffer, error, index, batch_size, num_frame)

        if buffer.num_blocks > 0:
            from wan_5b.distributed.sp_training import get_data_parallel_group
            comm_group = get_data_parallel_group()
            if comm_group is None:
                return self._collect_local_items(buffer, error, index, batch_size, num_frame)
            comm_size = dist.get_world_size(comm_group)
        else:
            comm_group = None
            comm_size = dist.get_world_size()

        if comm_size <= 1:
            return self._collect_local_items(buffer, error, index, batch_size, num_frame)

        err_local = error.detach().contiguous()
        idx_local = index.detach().contiguous()
        err_list = [torch.empty_like(err_local) for _ in range(comm_size)]
        idx_list = [torch.empty_like(idx_local) for _ in range(comm_size)]
        if comm_group is None:
            dist.all_gather(err_list, err_local)
            dist.all_gather(idx_list, idx_local)
        else:
            dist.all_gather(err_list, err_local, group=comm_group)
            dist.all_gather(idx_list, idx_local, group=comm_group)

        block_size = self.num_frame_per_block
        num_blocks = num_frame // block_size
        items = []
        for err_r, idx_r in zip(err_list, idx_list):
            idx_per_block = idx_r[:, ::block_size]
            err_blocks = err_r.reshape(
                batch_size, num_blocks, block_size, *err_r.shape[2:]
            )
            for b in range(batch_size):
                for blk in range(num_blocks):
                    pos = blk if buffer.num_blocks > 0 else None
                    items.append((err_blocks[b, blk], idx_per_block[b, blk].item(), pos))
        return items

    def _collect_local_items(self, buffer, error, index, batch_size, num_frame):
        """Same item-list format as ``_gather_errors_for_buffer`` but with no
        collective — used outside the warmup window or when distributed is off."""
        block_size = self.num_frame_per_block
        num_blocks = num_frame // block_size
        idx_per_block = index[:, ::block_size]
        error_blocks = error.reshape(
            batch_size, num_blocks, block_size, *error.shape[2:]
        )
        items = []
        for b in range(batch_size):
            for blk in range(num_blocks):
                pos = blk if buffer.num_blocks > 0 else None
                items.append((error_blocks[b, blk], idx_per_block[b, blk].item(), pos))
        return items

    def _apply_gathered_items(self, buffer, items):
        """Pure local: drop ``items`` into ``buffer``. No collective, no
        cross-rank coordination — each rank may invoke this independently
        (or skip it entirely) without risking a deadlock."""
        for err_block, t_idx, pos in items:
            buffer.add(err_block, t_idx, block_pos=pos)

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = CausalDiffusionInferencePipeline(
            args=self.args,
            device=self.device,
            generator=self.generator,
            text_encoder=self.text_encoder,
            vae=self.vae
        )
