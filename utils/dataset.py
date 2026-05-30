# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import Dataset
import numpy as np
import torch
import random
import json
from pathlib import Path
from PIL import Image
import os
import subprocess
import time
import warnings
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

try:
    import decord
except ModuleNotFoundError:
    decord = None

DEFAULT_SCENE_CUT_PREFIX = "The scene transitions. "


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class MultiTextDataset(Dataset):
    """Dataset for multi-segment prompts stored in a JSONL file.

    Each line is a JSON object, e.g.
        {"prompts": ["a cat", "a dog", "a bird"]}

    Args
    ----
    prompt_path : str
        Path to the JSONL file
    field       : str
        Name of the list-of-strings field, default "prompts"
    cache_dir   : str | None
        ``cache_dir`` passed to HF Datasets (optional)
    """

    def __init__(self, prompt_path: str, field: str = "prompts", cache_dir: str | None = None):
        try:
            import datasets
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The 'datasets' package is required for MultiTextDataset. "
                "Use MultiTextConcatDataset for plain txt/json-caption directories "
                "or install datasets."
            ) from exc
        self.ds = datasets.load_dataset(
            "json",
            data_files=prompt_path,
            split="train",
            cache_dir=cache_dir,
            streaming=False, 
        )

        assert len(self.ds) > 0, "JSONL is empty"
        assert field in self.ds.column_names, f"Missing field '{field}'"

        seg_len = len(self.ds[0][field])
        for i, ex in enumerate(self.ds):
            val = ex[field]
            assert isinstance(val, list), f"Line {i} field '{field}' is not a list"
            assert len(val) == seg_len,  f"Line {i} list length mismatch"

        self.field = field

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {
            "idx": idx,
            "prompts_list": self.ds[idx][self.field],  # List[str]
        }


class MultiTextConcatDataset(Dataset):
    """Text-only dataset for multi-shot training and inference.

    Supports two input modes:

    **txt file** — each line is one caption. Each sample uses the caption at
    index ``idx``, repeated ``num_blocks`` times (single-shot, no scene cut
    prefix).

    **directory** — reads ``caption/<subfolder>/*.json`` files (no video dir
    needed). Shot durations are resolved with a three-level fallback:

    1. ``shot_durations.txt`` in the caption subfolder (per-sample override)
    2. ``chunks_per_shot`` from config (global fixed repeat)
    3. Even distribution across all available captions

    Scene cut prefix is prepended at shot boundaries (first block of each
    shot except shot 0). Output is always exactly ``num_blocks`` prompts:
    truncated if too many, padded with the last caption if too few.
    """

    def __init__(
        self,
        data_path: str,
        num_blocks: int,
        chunks_per_shot: int = 0,
        scene_cut_prefix: str = DEFAULT_SCENE_CUT_PREFIX,
        caption_field: str = "caption",
        deterministic: bool = False,
    ):
        self.num_blocks = num_blocks
        self.chunks_per_shot = chunks_per_shot
        self.scene_cut_prefix = scene_cut_prefix
        self.caption_field = caption_field
        self.deterministic = deterministic

        path = Path(data_path)
        if data_path.endswith(".txt") or path.is_file():
            self._mode = "txt"
            with open(data_path, encoding="utf-8") as f:
                self._prompts = [line.rstrip() for line in f if line.strip()]
            assert len(self._prompts) > 0, f"No prompts found in {data_path}"
        else:
            self._mode = "dir"
            self._caption_dir = path / "caption" if (path / "caption").is_dir() else path
            self._folders = sorted([d for d in self._caption_dir.iterdir() if d.is_dir()])
            assert len(self._folders) > 0, (
                f"No caption subfolders found in {self._caption_dir}"
            )

    def __len__(self):
        if self._mode == "txt":
            return len(self._prompts)
        return len(self._folders)

    def __getitem__(self, idx):
        if self._mode == "txt":
            return self._get_txt_item(idx)
        return self._get_dir_item(idx)

    # ------------------------------------------------------------------
    # txt mode
    # ------------------------------------------------------------------

    def _get_txt_item(self, idx):
        caption = self._prompts[idx % len(self._prompts)]
        return {
            "prompts": [caption] * self.num_blocks,
            "idx": idx,
        }

    # ------------------------------------------------------------------
    # directory mode
    # ------------------------------------------------------------------

    def _get_dir_item(self, idx):
        folder = self._folders[idx % len(self._folders)]
        raw_captions = self._load_captions_from_folder(folder)
        if not raw_captions:
            raw_captions = [""]

        shot_durations = self._resolve_shot_durations(folder, len(raw_captions))
        prompts = self._apply_shot_durations(raw_captions, shot_durations)

        # Ensure exactly num_blocks prompts
        if len(prompts) > self.num_blocks:
            prompts = prompts[: self.num_blocks]
        elif len(prompts) < self.num_blocks:
            last = prompts[-1] if prompts else ""
            prompts.extend([last] * (self.num_blocks - len(prompts)))

        return {
            "prompts": prompts,
            "idx": idx,
        }

    def _load_captions_from_folder(self, folder: Path):
        json_files = sorted(
            [f for f in folder.glob("*.json") if f.name != "global.json"],
            key=lambda p: (p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 0, p.stem),
        )
        captions = []
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    captions.append(data.get(self.caption_field, ""))
            except Exception:
                captions.append("")
        return captions

    # ------------------------------------------------------------------
    # shot duration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_shot_durations(folder: Path):
        txt_path = folder / "shot_durations.txt"
        if not txt_path.exists():
            return None
        try:
            with open(txt_path, "r") as f:
                content = f.read().strip()
            parts = content.replace(",", " ").split()
            durations = [int(x) for x in parts if x.strip()]
            return durations if durations else None
        except Exception:
            return None

    def _resolve_shot_durations(self, folder: Path, num_captions: int):
        durations = self._load_shot_durations(folder)
        if durations is not None:
            return durations[:num_captions]
        if self.chunks_per_shot > 0:
            return [self.chunks_per_shot] * num_captions
        return self._even_durations(num_captions)

    def _even_durations(self, num_shots: int):
        total = self.num_blocks
        base, extra = divmod(total, num_shots)
        return [base + (1 if i < extra else 0) for i in range(num_shots)]

    def _apply_shot_durations(self, raw_captions, shot_durations):
        target = self.num_blocks
        clamped: list[int] = []
        remaining = target
        for d in shot_durations:
            if remaining <= 0:
                break
            take = min(d, remaining)
            clamped.append(take)
            remaining -= take
        if remaining > 0 and clamped:
            clamped[-1] += remaining

        prompts: list[str] = []
        for shot_idx, (caption, duration) in enumerate(zip(raw_captions, clamped)):
            for block_in_shot in range(duration):
                if shot_idx > 0 and block_in_shot == 0 and self.scene_cut_prefix:
                    prompts.append(self.scene_cut_prefix + caption)
                else:
                    prompts.append(caption)
        return prompts


class MultiVideoConcatDataset(Dataset):
    """Dataset that concatenates multiple videos from a folder into a fixed-length video.
    
    Each item consists of multiple video segments concatenated together:
    - First segment: first_chunk_frames frames (chunk)
    - Subsequent segments: subsequent_chunk_frames frames each (chunk)
    - Total: total_frames frames (first_chunk_frames + subsequent_chunk_frames*num_subsequent_segments = total_frames)
    
    Videos are sampled preserving original duration, and if a video doesn't have enough frames,
    it moves to the next video. If a video has enough frames, it can be sampled repeatedly.
    """
    def __init__(
        self,
        data_dir,
        video_size,
        total_frames,
        target_fps=16,
        video_extensions=('.mp4', '.avi', '.mov', '.mkv', '.webm'),
        caption_field='caption',
        filter_invalid_folders=False,
        deterministic: bool = False,
        num_frame_per_block=8,
        temporal_compression_ratio=4,
        allow_padding: bool = False,
        min_latent_frames: int = 0,
        single_video_only: bool = False,
        independent_first_frame: bool = False,
        return_image: bool = False,
        max_chunks_per_shot: int = 0,
        scene_cut_prefix: str = DEFAULT_SCENE_CUT_PREFIX,
        sample_warning_seconds: float = 60.0,
        sample_warning_interval_seconds: float = 60.0,
    ):
        self.root_dir = Path(data_dir)
        self.data_dir = self.root_dir / "video"
        self.caption_dir = self.root_dir / "caption"
        self.video_size = video_size
        self.total_frames = total_frames

        total_latent_frames = 1 + (total_frames - 1) // temporal_compression_ratio
        separate_first_latent = (
            independent_first_frame
            and total_latent_frames % num_frame_per_block != 0
        )
        if separate_first_latent:
            assert (total_latent_frames - 1) % num_frame_per_block == 0, (
                f"total latent frames ({total_latent_frames}) must be divisible by "
                f"num_frame_per_block ({num_frame_per_block}) or equal to "
                f"1 + N * num_frame_per_block when independent_first_frame=True"
            )
        first_chunk_latent_frames = (
            num_frame_per_block + 1 if separate_first_latent else num_frame_per_block
        )
        first_chunk_frames = 1 + (first_chunk_latent_frames - 1) * temporal_compression_ratio
        subsequent_chunk_frames = num_frame_per_block * temporal_compression_ratio

        self.first_chunk_frames = first_chunk_frames
        self.subsequent_chunk_frames = subsequent_chunk_frames
        self.target_fps = target_fps
        self.caption_field = caption_field
        self.video_extensions = video_extensions
        self.filter_invalid_folders = filter_invalid_folders
        self.deterministic = deterministic
        self.allow_padding = allow_padding
        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = independent_first_frame
        self.first_chunk_latent_frames = first_chunk_latent_frames
        self.return_image = return_image
        if min_latent_frames > 0:
            assert min_latent_frames % num_frame_per_block == 0, (
                f"min_latent_frames ({min_latent_frames}) must be a multiple of "
                f"num_frame_per_block ({num_frame_per_block})"
            )
        self.min_latent_frames = min_latent_frames
        self.single_video_only = single_video_only
        self.max_chunks_per_shot = max_chunks_per_shot
        self.scene_cut_prefix = scene_cut_prefix
        self.sample_warning_seconds = float(sample_warning_seconds or 0.0)
        self.sample_warning_interval_seconds = float(sample_warning_interval_seconds or 0.0)
        
        remaining_frames = total_frames - first_chunk_frames
        self.num_subsequent_segments = remaining_frames // subsequent_chunk_frames
        self.total_segments = 1 + self.num_subsequent_segments
        
        assert total_frames == first_chunk_frames + self.num_subsequent_segments * subsequent_chunk_frames, \
            f"Total frames ({total_frames}) must equal first_chunk_frames ({first_chunk_frames}) + " \
            f"num_subsequent_segments ({self.num_subsequent_segments}) * subsequent_chunk_frames ({subsequent_chunk_frames})"
        
        if not self.data_dir.exists():
            raise ValueError(f"Video directory not found: {self.data_dir}")
        if not self.caption_dir.exists():
            raise ValueError(f"Caption directory not found: {self.caption_dir}")
        
        self.folders = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if len(self.folders) == 0:
            raise ValueError(f"No subdirectories found in {self.data_dir}")

        # Optionally pre-filter folders with insufficient frames
        # Note: This can be slow for large datasets due to IO operations
        if self.filter_invalid_folders:
            print(f"[MultiVideoConcatDataset] Pre-filtering {len(self.folders)} folders for sufficient frames...")
            valid_folders = []
            skipped_folders = []
            for folder in self.folders:
                if self._check_folder_has_enough_frames(folder):
                    valid_folders.append(folder)
                else:
                    skipped_folders.append(folder.name)
            
            if len(skipped_folders) > 0:
                print(f"[MultiVideoConcatDataset] Skipped {len(skipped_folders)} folders due to insufficient frames: {skipped_folders}")
            
            self.folders = valid_folders
            
            if len(self.folders) == 0:
                raise ValueError(f"No folders with sufficient frames found in {self.data_dir}")
        
        # Setup transforms
        self.resize_transform = transforms.Resize(
            self.video_size, 
            antialias=True
        )
        self.normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        
        # Lazy caches: avoid repeated decord.VideoReader / filesystem IO
        self._video_info_cache = {}   # video_path -> (total_frames, fps)
        self._folder_files_cache = {} # folder_path -> list of video paths
        
        if decord is not None:
            decord.bridge.set_bridge('torch')
    
    def _get_caption_folder(self, folder_name):
        """Return the caption directory path for a given folder (sample)."""
        return self.caption_dir / folder_name

    def _load_caption(self, video_path, folder_name):
        """Load caption for a video file."""
        video_stem = video_path.stem
        caption_folder = self._get_caption_folder(folder_name)
        caption_path = caption_folder / f"{video_stem}.json"
        
        if caption_path.exists():
            try:
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption_data = json.load(f)
                    return caption_data.get(self.caption_field, "")
            except Exception:
                return ""
        return ""

    def _get_video_files_in_folder(self, folder_path):
        """Get sorted video files in a folder, keeping only those with a per-video caption (cached)."""
        key = str(folder_path)
        if key in self._folder_files_cache:
            return self._folder_files_cache[key]

        video_files = []
        for ext in self.video_extensions:
            video_files.extend(list(folder_path.glob(f'*{ext}')))
            video_files.extend(list(folder_path.glob(f'*{ext.upper()}')))
        
        def get_numeric_key(path):
            try:
                return int(path.stem)
            except ValueError:
                return float('inf')
        
        video_files.sort(key=get_numeric_key)

        folder_name = folder_path.name
        filtered_videos = []
        for video_path in video_files:
            caption = self._load_caption(video_path, folder_name)
            if caption is not None and caption != "":
                filtered_videos.append(video_path)

        self._folder_files_cache[key] = filtered_videos
        return filtered_videos
    
    def _check_folder_has_enough_frames(self, folder_path):
        """Check if a folder has enough total frames across all videos to complete all segments.
        
        This is a lenient check: we verify that the total available frames across all videos
        is sufficient for the required segments, assuming ideal sampling.
        """
        video_files = self._get_video_files_in_folder(folder_path)
        if len(video_files) == 0:
            return False
        
        # Calculate total available frames (in target_fps timebase)
        total_available_frames = 0
        for video_path in video_files:
            try:
                total_frames, original_fps = self._get_video_info(video_path)
                # Convert to target_fps timebase
                available_in_target_fps = total_frames * self.target_fps / original_fps
                total_available_frames += available_in_target_fps
            except Exception:
                # If we can't read a video, be conservative and skip this folder
                return False
        
        # Check if we have enough frames for all segments
        required_frames = self.total_frames
        return total_available_frames >= required_frames
    
    def _get_video_info(self, video_path):
        """Get video information without loading frames (cached)."""
        key = str(video_path)
        if key in self._video_info_cache:
            return self._video_info_cache[key]
        if decord is None:
            info = self._get_video_info_ffprobe(video_path)
            self._video_info_cache[key] = info
            return info
        try:
            vr = decord.VideoReader(key, width=self.video_size[1], height=self.video_size[0])
        except:
            vr = decord.VideoReader(key)
        info = (len(vr), vr.get_avg_fps())
        self._video_info_cache[key] = info
        return info

    @staticmethod
    def _parse_fps(value):
        if not value or value == "0/0":
            return 0.0
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f != 0 else 0.0
        return float(value)

    def _get_video_info_ffprobe(self, video_path):
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames,avg_frame_rate,r_frame_rate,duration",
            "-of",
            "json",
            str(video_path),
        ]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            stream = json.loads(proc.stdout)["streams"][0]
            fps = self._parse_fps(stream.get("avg_frame_rate")) or self._parse_fps(stream.get("r_frame_rate"))
            frames_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
            if frames_raw and str(frames_raw).isdigit():
                total_frames = int(frames_raw)
            else:
                total_frames = int(round(float(stream.get("duration", 0.0)) * fps))
            if total_frames <= 0 or fps <= 0:
                raise ValueError(f"Could not infer frames/fps from ffprobe output for {video_path}")
            return total_frames, fps
        except (subprocess.CalledProcessError, KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"ffprobe failed to read video metadata for {video_path}: {exc}") from exc
    
    def _can_sample_from_position(self, total_frames, original_fps, num_frames, start_frame):
        """Check if we can sample num_frames starting from start_frame.
        
        Returns True if we can sample num_frames without exceeding video bounds.
        """
        if start_frame >= total_frames:
            return False
        
        # Calculate sampling interval: original_fps / target_fps
        sampling_interval = original_fps / self.target_fps
        
        # Calculate the last frame index we need (with rounding)
        # We need to check if we can get num_frames frames
        last_frame_needed = start_frame + (num_frames - 1) * sampling_interval
        
        # Account for rounding: the actual last frame index will be rounded
        # So we need some margin to ensure we don't exceed bounds
        return int(np.round(last_frame_needed)) < total_frames

    def _can_complete_all_segments_without_wrap(self, video_files, start_video_idx, start_frame):
        """Check if from (start_video_idx, start_frame) we can sample all segments
        without ever wrapping to the beginning (i.e. only use this video and later ones).
        When single_video_only is True, all segments must come from the same video.
        """
        vidx = start_video_idx
        start = start_frame

        # First segment
        total_frames, original_fps = self._get_video_info(video_files[vidx])
        if not self._can_sample_from_position(
            total_frames, original_fps, self.first_chunk_frames, start
        ):
            return False
        sampling_interval = original_fps / self.target_fps
        next_start = start + self.first_chunk_frames * sampling_interval
        if int(np.round(next_start)) >= total_frames:
            if self.single_video_only:
                return self.num_subsequent_segments == 0
            vidx += 1
            start = 0
        else:
            start = int(np.round(next_start))
        if vidx >= len(video_files):
            return False

        # Subsequent segments
        for seg_i in range(self.num_subsequent_segments):
            while vidx < len(video_files):
                total_frames, original_fps = self._get_video_info(video_files[vidx])
                if self._can_sample_from_position(
                    total_frames, original_fps, self.subsequent_chunk_frames, start
                ):
                    break
                if self.single_video_only:
                    return False
                vidx += 1
                start = 0
            if vidx >= len(video_files):
                return False
            sampling_interval = original_fps / self.target_fps
            next_start = start + self.subsequent_chunk_frames * sampling_interval
            if int(np.round(next_start)) >= total_frames:
                if self.single_video_only and seg_i < self.num_subsequent_segments - 1:
                    return False
                vidx += 1
                start = 0
            else:
                start = int(np.round(next_start))
        return True

    def _sample_random_start(self, video_files):
        """Sample a random (video_idx, start_frame) valid for the first segment,
        and from which we can complete ALL segments without wrapping to the start.
        Returns (video_idx, start_frame); falls back to (0, 0) if no valid start found.
        """
        candidates = []
        for video_idx, video_path in enumerate(video_files):
            total_frames, original_fps = self._get_video_info(video_path)
            sampling_interval = original_fps / self.target_fps
            last_needed = (self.first_chunk_frames - 1) * sampling_interval
            max_start = int(np.floor(total_frames - 1 - last_needed))
            if max_start < 0:
                continue
            step = max(1, max_start // 50)
            for start in range(0, max_start + 1, step):
                if not self._can_sample_from_position(
                    total_frames, original_fps, self.first_chunk_frames, start
                ):
                    continue
                if self._can_complete_all_segments_without_wrap(video_files, video_idx, start):
                    candidates.append((video_idx, start))
        if candidates:
            chosen = random.choice(candidates)
            return chosen
        return (0, 0)
    
    def _sample_frames_from_video(self, video_path, num_frames, start_frame=0):
        """Sample frames from a video preserving original duration.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample
            start_frame: Starting frame index (for repeated sampling)
        
        Returns:
            tuple: (frames_tensor, total_frames_in_video, original_fps)
        """
        if decord is None:
            return self._sample_frames_from_video_ffmpeg(video_path, num_frames, start_frame)

        try:
            vr = decord.VideoReader(str(video_path), width=self.video_size[1], height=self.video_size[0])
        except:
            vr = decord.VideoReader(str(video_path))
        
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()
        
        if total_frames == 0:
            raise ValueError(f"Video {video_path} has no frames")
        
        # Calculate frame sampling based on fps to preserve duration
        # Calculate sampling interval: original_fps / target_fps
        sampling_interval = original_fps / self.target_fps
        
        # Generate frame indices starting from start_frame
        indices = []
        current_frame = float(start_frame)
        for _ in range(num_frames):
            frame_idx = int(np.round(current_frame))
            frame_idx = min(frame_idx, total_frames - 1)
            indices.append(frame_idx)
            current_frame += sampling_interval
            if current_frame >= total_frames:
                # If we run out of frames, pad with the last frame
                remaining = num_frames - len(indices)
                indices.extend([total_frames - 1] * remaining)
                break
        
        indices = np.array(indices[:num_frames], dtype=np.int32)
        
        # Get video frames: shape (num_frames, height, width, 3)
        video_frames = vr.get_batch(indices).numpy()
        
        # Convert to tensor and permute to (num_frames, 3, height, width)
        video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).contiguous()
        
        # Convert to float and normalize pixel values from [0, 255] to [0, 1]
        video_tensor = video_tensor.float() / 255.0
        
        # Resize if needed
        if video_tensor.shape[2] != self.video_size[0] or video_tensor.shape[3] != self.video_size[1]:
            resized_frames = []
            for i in range(video_tensor.shape[0]):
                resized_frames.append(self.resize_transform(video_tensor[i]))
            video_tensor = torch.stack(resized_frames, dim=0)
        
        # Apply normalization: (x - 0.5) / 0.5 -> range [-1, 1]
        video_tensor = self.normalize(video_tensor)
        video_tensor = video_tensor.to(torch.float16)
        
        return video_tensor, total_frames, original_fps

    def _sample_frames_from_video_ffmpeg(self, video_path, num_frames, start_frame=0):
        total_frames, original_fps = self._get_video_info(video_path)
        start_time = max(0.0, float(start_frame) / max(original_fps, 1e-6))
        height, width = self.video_size
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start_time:.6f}",
            "-i",
            str(video_path),
            "-vf",
            f"fps={self.target_fps},scale={width}:{height}",
            "-frames:v",
            str(num_frames),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            raise RuntimeError(f"ffmpeg failed to sample {video_path}: {stderr}") from exc

        frame_bytes = height * width * 3
        decoded_frames = len(proc.stdout) // frame_bytes
        if decoded_frames <= 0:
            raise RuntimeError(f"ffmpeg produced no frames for {video_path}")

        video_frames = np.frombuffer(proc.stdout[: decoded_frames * frame_bytes], dtype=np.uint8)
        video_frames = video_frames.reshape(decoded_frames, height, width, 3)
        video_tensor = torch.from_numpy(video_frames.copy()).permute(0, 3, 1, 2).contiguous()
        video_tensor = video_tensor.float() / 255.0
        if decoded_frames < num_frames:
            pad = video_tensor[-1:].repeat(num_frames - decoded_frames, 1, 1, 1)
            video_tensor = torch.cat([video_tensor, pad], dim=0)
        elif decoded_frames > num_frames:
            video_tensor = video_tensor[:num_frames]

        video_tensor = self.normalize(video_tensor)
        video_tensor = video_tensor.to(torch.float16)
        return video_tensor, total_frames, original_fps
    
    def __len__(self):
        return len(self.folders)
    
    @staticmethod
    def _sample_failure(reason):
        return False, reason

    def _try_get_item_from_folder(self, folder_idx, deterministic: bool = False):
        """Try to get item from a specific folder.

        Returns (True, result) on success and (False, reason) on failure.
        The reason is used by __getitem__ warnings so a long scan does not
        look like a silent hang when most folders are too short.
        """
        folder_path = self.folders[folder_idx]
        folder_name = folder_path.name
        
        video_files = self._get_video_files_in_folder(folder_path)
        
        if len(video_files) == 0:
            return self._sample_failure(f"{folder_name}: no videos with captions")
        
        # Collect all segments
        all_segments = []
        prompts_list = []
        
        try:
            # Start position
            if deterministic:
                current_video_idx, current_start_frame = 0, 0
            else:
                # Random start: don't always start from the first video, so we get more diversity
                current_video_idx, current_start_frame = self._sample_random_start(video_files)
            
            # Sample first segment (9 frames)
            video_path = video_files[current_video_idx]
            prev_seg_video_idx = current_video_idx
            total_frames, original_fps = self._get_video_info(video_path)

            # Ensure the current video is long enough for the first segment
            while not self._can_sample_from_position(
                total_frames, original_fps, self.first_chunk_frames, current_start_frame
            ):
                if self.single_video_only:
                    return self._sample_failure(
                        f"{folder_name}: first video is too short for the first chunk"
                    )
                current_video_idx += 1
                current_start_frame = 0
                if current_video_idx >= len(video_files):
                    return self._sample_failure(
                        f"{folder_name}: no video can provide the first chunk"
                    )
                video_path = video_files[current_video_idx]
                prev_seg_video_idx = current_video_idx
                total_frames, original_fps = self._get_video_info(video_path)

            # Sample first segment
            segment_frames, total_frames, original_fps = self._sample_frames_from_video(
                video_path, self.first_chunk_frames, current_start_frame
            )
            all_segments.append(segment_frames)
            
            prompt = self._load_caption(video_path, folder_name)
            prompts_list.append(prompt)
            
            # Update position for next sampling
            sampling_interval = original_fps / self.target_fps
            next_start_frame = current_start_frame + self.first_chunk_frames * sampling_interval
            
            # If we've exhausted this video, move to next
            if int(np.round(next_start_frame)) >= total_frames:
                if self.single_video_only:
                    if self.num_subsequent_segments > 0:
                        return self._sample_failure(
                            f"{folder_name}: single-video sample ends after first chunk"
                        )
                else:
                    current_video_idx += 1
                    current_start_frame = 0
            else:
                current_start_frame = int(np.round(next_start_frame))

            chunks_from_current_video = 1 if current_video_idx == prev_seg_video_idx else 0

            # Sample subsequent segments (12 frames each). No wrap: we only use videos from start onward.
            for seg_idx in range(self.num_subsequent_segments):
                if current_video_idx >= len(video_files):
                    if self.allow_padding:
                        break
                    return self._sample_failure(
                        f"{folder_name}: ran out of videos before all chunks were filled"
                    )

                # Force virtual scene cut if max_shot_chunks reached:
                # skip 1 second of video and treat the remainder as a new shot.
                forced_scene_cut = False
                if (self.max_chunks_per_shot > 0
                        and chunks_from_current_video >= self.max_chunks_per_shot):
                    vp = video_files[current_video_idx]
                    _, ofps = self._get_video_info(vp)
                    current_start_frame += int(np.round(ofps))
                    chunks_from_current_video = 0
                    forced_scene_cut = True

                video_path = video_files[current_video_idx]
                total_frames, original_fps = self._get_video_info(video_path)

                can_sample = True
                while not self._can_sample_from_position(
                    total_frames, original_fps, self.subsequent_chunk_frames, current_start_frame
                ):
                    if self.single_video_only:
                        can_sample = False
                        break
                    current_video_idx += 1
                    current_start_frame = 0
                    chunks_from_current_video = 0
                    if current_video_idx >= len(video_files):
                        can_sample = False
                        break
                    video_path = video_files[current_video_idx]
                    total_frames, original_fps = self._get_video_info(video_path)

                if not can_sample:
                    if self.allow_padding:
                        break
                    return self._sample_failure(
                        f"{folder_name}: remaining videos are too short for the next chunk"
                    )

                is_scene_cut = (current_video_idx != prev_seg_video_idx) or forced_scene_cut

                # Sample segment
                segment_frames, total_frames, original_fps = self._sample_frames_from_video(
                    video_path, self.subsequent_chunk_frames, current_start_frame
                )
                all_segments.append(segment_frames)
                
                prompt = self._load_caption(video_path, folder_name)
                if is_scene_cut and self.scene_cut_prefix:
                    prompt = self.scene_cut_prefix + prompt
                prompts_list.append(prompt)

                prev_seg_video_idx = current_video_idx
                chunks_from_current_video += 1

                # Update position for next sampling
                sampling_interval = original_fps / self.target_fps
                next_start_frame = current_start_frame + self.subsequent_chunk_frames * sampling_interval
                
                # If we've exhausted this video, move to next
                if int(np.round(next_start_frame)) >= total_frames:
                    if self.single_video_only:
                        if seg_idx < self.num_subsequent_segments - 1:
                            return self._sample_failure(
                                f"{folder_name}: single-video sample is too short"
                            )
                    else:
                        current_video_idx += 1
                        current_start_frame = 0
                        chunks_from_current_video = 0
                else:
                    current_start_frame = int(np.round(next_start_frame))

            num_filled_segments = len(all_segments)
            if num_filled_segments == 0:
                num_valid_latent_frames = 0
            else:
                num_valid_latent_frames = (
                    self.first_chunk_latent_frames
                    + (num_filled_segments - 1) * self.num_frame_per_block
                )

            # Reject if below minimum latent frame threshold
            if self.allow_padding and self.min_latent_frames > 0:
                if num_valid_latent_frames < self.min_latent_frames:
                    return self._sample_failure(
                        f"{folder_name}: only {num_valid_latent_frames} valid latent frames, "
                        f"below min_latent_frames={self.min_latent_frames}"
                    )

            if num_filled_segments < self.total_segments:
                last_prompt = prompts_list[-1] if prompts_list else ""
                prompts_list.extend([last_prompt] * (self.total_segments - num_filled_segments))
            
            # Concatenate all segments: (total_frames, 3, height, width)
            concatenated_video = torch.cat(all_segments, dim=0)
            
            # Ensure we have exactly total_frames
            if concatenated_video.shape[0] != self.total_frames:
                # Pad or trim if necessary
                if concatenated_video.shape[0] < self.total_frames:
                    # Pad with last frame
                    last_frame = concatenated_video[-1:].repeat(self.total_frames - concatenated_video.shape[0], 1, 1, 1)
                    concatenated_video = torch.cat([concatenated_video, last_frame], dim=0)
                else:
                    # Trim
                    concatenated_video = concatenated_video[:self.total_frames]
            
            result = {
                'frames': concatenated_video.permute(1, 0, 2, 3),
                'prompts': prompts_list,
                'idx': folder_idx
            }
            if self.return_image:
                result['image'] = concatenated_video[0]
            if self.allow_padding:
                result['num_valid_latent_frames'] = num_valid_latent_frames
            return True, result
        except Exception as exc:
            return self._sample_failure(f"{folder_name}: {type(exc).__name__}: {exc}")

    def __getitem__(self, idx):
        start_time = time.monotonic()
        last_warning_time = start_time
        attempts = 0
        last_failure = None

        def maybe_warn(folder_idx, failure_reason):
            nonlocal last_warning_time
            if self.sample_warning_seconds <= 0:
                return
            elapsed = time.monotonic() - start_time
            if elapsed < self.sample_warning_seconds:
                return
            if (
                self.sample_warning_interval_seconds > 0
                and time.monotonic() - last_warning_time < self.sample_warning_interval_seconds
            ):
                return
            last_warning_time = time.monotonic()
            folder_name = self.folders[folder_idx % len(self.folders)].name
            warnings.warn(
                "[MultiVideoConcatDataset] Still searching for a valid sample "
                f"after {elapsed:.1f}s and {attempts} folder attempts "
                f"(requested_idx={idx}, current_folder={folder_name}, "
                f"last_failure={failure_reason}). This usually means the dataset "
                f"does not contain enough video duration for total_frames={self.total_frames} "
                f"at target_fps={self.target_fps}. Consider reducing the training "
                "window, enabling allow_padding, lowering min_latent_frames, or "
                "pre-filtering invalid folders.",
                RuntimeWarning,
                stacklevel=2,
            )

        # First try the requested folder
        attempts += 1
        success, result = self._try_get_item_from_folder(idx, deterministic=self.deterministic)
        if success:
            return result
        last_failure = result
        maybe_warn(idx, last_failure)
        
        # If the requested folder fails, try other folders.
        # If any valid folder exists in the dataset, try to return data from it:
        # scan every other folder starting from idx + 1 and return the first
        # successful sample.
        num_folders = len(self.folders)
        for i in range(1, num_folders):
            alt_idx = (idx + i) % num_folders
            attempts += 1
            success, result = self._try_get_item_from_folder(
                alt_idx,
                deterministic=self.deterministic,
            )
            if success:
                return result
            last_failure = result
            maybe_warn(alt_idx, last_failure)
        
        # If all attempts fail, raise an error
        elapsed = time.monotonic() - start_time
        if self.sample_warning_seconds > 0 and elapsed >= self.sample_warning_seconds:
            warnings.warn(
                "[MultiVideoConcatDataset] No valid sample was found after "
                f"{elapsed:.1f}s and {attempts} folder attempts "
                f"(requested_idx={idx}, last_failure={last_failure}).",
                RuntimeWarning,
                stacklevel=2,
            )
        raise ValueError(
            f"Failed to sample valid data from folder {idx} and nearby folders. "
            f"This may indicate insufficient valid videos in the dataset. "
            f"Tried {attempts} folders in {elapsed:.1f}s. Last failure: {last_failure}"
        )


def cycle(dl):
    while True:
        for data in dl:
            yield data

def multi_video_collate_fn(batch):
    # batch is a length-B list of dictionaries returned by __getitem__.
    frames = torch.stack([b["frames"] for b in batch], dim=0)  # (B, T, C, H, W)

    # Keep prompts as one list per sample:
    # [[p0_seg0, p0_seg1, ...], [p1_seg0, ...], ...].
    prompts_list = [b["prompts"] for b in batch]          # List[List[str]]

    idx = torch.tensor([b["idx"] for b in batch], dtype=torch.long)

    result = {
        "frames": frames,
        "prompts": prompts_list,
        "idx": idx,
    }

    if "image" in batch[0]:
        result["image"] = torch.stack([b["image"] for b in batch], dim=0)

    if "num_valid_latent_frames" in batch[0]:
        result["num_valid_latent_frames"] = torch.tensor(
            [b["num_valid_latent_frames"] for b in batch], dtype=torch.long
        )

    return result


def eval_collate_fn(batch):
    """Collate for text-only datasets (no frames)."""
    prompts_list = [b["prompts"] for b in batch]
    idx = torch.tensor([b["idx"] for b in batch], dtype=torch.long)
    result = {
        "prompts": prompts_list,
        "idx": idx,
    }
    if "shot_durations" in batch[0]:
        result["shot_durations"] = [b["shot_durations"] for b in batch]
    return result
