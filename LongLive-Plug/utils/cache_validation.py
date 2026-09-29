"""Verify CFG cache shards once across training ranks before allocating models."""
import hashlib
import json
import os
from pathlib import Path

import torch.distributed as dist


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_distributed_cache(manifest_path):
    path = Path(manifest_path).resolve()
    rank, world = dist.get_rank(), dist.get_world_size()
    error = None
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
        shards = {}
        for record in manifest['records']:
            shard = record['shard']
            expected = record['shard_sha256']
            if shard in shards and shards[shard] != expected:
                raise ValueError('Conflicting shard checksums')
            shards[shard] = expected
        for i, (relative, expected) in enumerate(sorted(shards.items())):
            if i % world != rank:
                continue
            target = (path.parent/relative).resolve()
            if not target.is_relative_to(path.parent) or sha256_file(target) != expected:
                raise ValueError(f'Cache shard checksum mismatch: {relative}')
        digest = hashlib.sha256(raw).hexdigest()
    except Exception as exc:
        error = str(exc)
        digest = None
    outcomes = [None] * world
    dist.all_gather_object(outcomes, {'error':error, 'manifest_sha256':digest})
    if any(x['error'] for x in outcomes) or len({x['manifest_sha256'] for x in outcomes}) != 1:
        raise RuntimeError(f'CFG cache verification failed: {outcomes}')
    os.environ['CFG_TEACHER_TRAJECTORY_VERIFIED_MANIFEST_SHA256'] = digest
