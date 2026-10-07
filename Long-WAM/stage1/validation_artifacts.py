#!/usr/bin/env python3
# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provenance: Original Long-WAM robot-video integration, adapted from the author branch.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/validation_artifacts.py
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

"""Decode/verify fixed validation videos; optionally upload to a NEW W&B run.

Never modifies historical runs or receipts. A validation-only generator load
uses internal step zero; --training-step comes from the bound portable launch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def verify(output):
    launches = sorted(output.glob('launch_*.json'))
    if not launches:
        raise ValueError('no bound portable launch')
    launch = json.loads(launches[-1].read_text())
    payload = dict(launch)
    expected = payload.pop('receipt_sha256')
    if hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != expected:
        raise ValueError('launch digest mismatch')
    cfg = launch['config']
    root = output / 'runtime/generated_video_000000'
    artifacts = []
    for group in cfg['evaluation']['groups']:
        group_dir = root / group['name']
        manifest = json.loads((group_dir / 'manifest.json').read_text())
        if (manifest['num_frames'] != group['num_frames'] or manifest['sampling_steps'] != 50 or
            manifest['guidance_scale'] != 3.0 or manifest['weight_modes'] != ['model'] or
            [s['sample_id'] for s in manifest['samples']] != group['sample_ids']):
            raise ValueError('fixed panel contract changed')
        for slot, sample in enumerate(manifest['samples']):
            # Same deterministic seed helper as the unchanged trainer.
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent / 'third_party/LongLive'))
            from utils.evaluation import deterministic_evaluation_seed
            if sample['slot'] != slot or sample['noise_seed'] != deterministic_evaluation_seed(20260711, slot):
                raise ValueError('slot/seed mismatch')
            video = group_dir / f'sample_{slot:02d}_{sample["sample_id"]}_model.mp4'
            prompt = group_dir / f'sample_{slot:02d}_{sample["sample_id"]}.txt'
            if sample['instruction'] != prompt.read_text().strip():
                raise ValueError('prompt mismatch')
            probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
                '-show_entries','stream=width,height,nb_read_frames,avg_frame_rate','-of','json',str(video)],
                capture_output=True, text=True, check=True)
            stream = json.loads(probe.stdout)['streams'][0]
            if (int(stream['nb_read_frames']) != 4*(group['num_frames']-1)+1 or
                (stream['width'],stream['height']) != (1280,704) or stream['avg_frame_rate'] != '24/1'):
                raise ValueError(f'wrong video geometry: {video}')
            subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(video),'-f','null','-'],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            artifacts.append(dict(group=group['name'],slot=slot,sample_id=sample['sample_id'],
                path=str(video),size_bytes=video.stat().st_size,sha256=sha(video),
                prompt=sample['instruction'],noise_seed=sample['noise_seed']))
    return launch, artifacts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--upload', action='store_true')
    p.add_argument('--entity')
    p.add_argument('--project')
    p.add_argument('--run-id', help='A new, explicit W&B run ID; never a historical production run.')
    a = p.parse_args()
    launch, artifacts = verify(a.output.resolve())
    print(json.dumps(dict(status='decoded_and_verified',stage=launch['stage'],
        training_step=launch['training_step'],videos=len(artifacts)),indent=2))
    if a.upload:
        if not a.entity or not a.project or not a.run_id:
            p.error('--upload requires --entity, --project and --run-id')
        import wandb
        # resume=never prevents an accidental append to a previous experiment.
        run = wandb.init(entity=a.entity,project=a.project,id=a.run_id,resume='never',
                         job_type='validation',mode='online',config=launch)
        try:
            for item in artifacts:
                run.log({f'validation/{item["group"]}/sample_{item["slot"]:02d}':
                         wandb.Video(item['path'],fps=24,format='mp4',caption=item['prompt'])},
                        step=launch['training_step'],commit=False)
            run.log({'validation/num_videos':len(artifacts)},step=launch['training_step'])
            manifest = wandb.Artifact('fixed-validation-manifest',type='validation-manifest')
            with manifest.new_file('videos.json',mode='w') as stream:
                json.dump(artifacts,stream,indent=2)
            run.log_artifact(manifest)
        finally:
            run.finish()


if __name__ == '__main__':
    main()
