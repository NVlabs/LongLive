"""Download pinned public assets and reproduce the training/heldout split."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.prompt_utils import FORMAT, normalize_prompt


def prepare_prompts(raw, provenance, output):
    if hashlib.sha256(raw).hexdigest() != provenance['prompt_sha256']:
        raise ValueError('Prompt source checksum mismatch')
    unique = list(dict.fromkeys(s.strip() for s in raw.decode().splitlines() if s.strip()))
    ordered = sorted(unique, key=lambda s: hashlib.sha256(('reproduce-20260927:' + s).encode()).hexdigest())
    heldout, selected = ordered[:16], ordered[16:80]
    train = [s for s in unique if s not in set(heldout)]
    if set(map(normalize_prompt, train)) & set(map(normalize_prompt, heldout)):
        raise ValueError('Normalized training/heldout prompt overlap')
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [('train', train), ('heldout', heldout), ('cfg_train_64', selected)]:
        (output / f'{name}.txt').write_text('\n'.join(values) + '\n')
    payload = {
        'schema_version': 1, 'format': FORMAT,
        'selection': {'selected_count': 64, 'algorithm': 'sha256(reproduce-20260927: + text), ranks 16..79'},
        'inputs': provenance,
        'outputs': {'prompt_file_sha256': hashlib.sha256((output/'cfg_train_64.txt').read_bytes()).hexdigest()},
        'records': [{'selected_index': i, 'prompt': p, 'prompt_sha256': hashlib.sha256(p.encode()).hexdigest(),
                     'normalized_prompt': normalize_prompt(p)} for i, p in enumerate(selected)],
    }
    (output/'cfg_train_64.provenance.json').write_text(json.dumps(payload, indent=2) + '\n')
    (output/'source_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--model-root', type=Path, default=Path('wan_models'))
    p.add_argument('--prompts-only', action='store_true')
    args = p.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    provenance = json.loads((ROOT/'docs/assets.json').read_text())
    path = hf_hub_download(provenance['prompt_repo'], provenance['prompt_file'],
                           revision=provenance['prompt_revision'])
    prepare_prompts(Path(path).read_bytes(), provenance, args.data_root)
    if not args.prompts_only:
        for name, model in provenance['models'].items():
            snapshot_download(model['repo'], revision=model['revision'], local_dir=args.model_root/name,
                              allow_patterns=['*.json', '*.safetensors', '*.pth', 'google/**'], max_workers=4)
    print('Pinned assets ready')


if __name__ == '__main__':
    main()
