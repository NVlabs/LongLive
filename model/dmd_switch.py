# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import time

from model.base import SelfForcingModel
from utils.memory import log_gpu_memory
import torch.distributed as dist
from model.dmd import DMD
from pipeline.streaming_switch_training import StreamingSwitchTrainingPipeline
from einops import rearrange

from utils.debug_option import DEBUG, LOG_GPU_MEMORY


class DMDSwitch(DMD):
    """DMD variant that supports mid-video prompt switching."""

    def _initialize_inference_pipeline(self):
        self.inference_pipeline = StreamingSwitchTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            context_noise=self.args.context_noise,
            local_attn_size=getattr(self.args, "model_kwargs", {}).get("local_attn_size", -1),
            slice_last_frames=getattr(self.args, "slice_last_frames", 21),
            global_sink=getattr(self.args, "global_sink", False),
        )