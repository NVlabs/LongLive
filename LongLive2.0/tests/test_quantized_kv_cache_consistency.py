import ast
import itertools
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F


def _load_cache_functions(outplace):
    """Load the real cache logic without the GPU/model-loading dependencies."""
    path = Path(__file__).resolve().parents[1] / "wan_5b/modules/causal_model.py"
    tree = ast.parse(path.read_text())
    attention_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CausalWanSelfAttention"
    )
    model_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CausalWanModel"
    )
    apply_updates = next(
        node for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_apply_cache_updates"
    )

    def attention(q, k, v):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    def rope(x, *args, **kwargs):
        # A deterministic rotation exercises both pre-cache and post-cache RoPE
        # placement. Positional embedding accuracy is outside this regression.
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)

    namespace = {
        "torch": torch,
        "nn": nn,
        "math": math,
        "attention": attention,
        "causal_rope_apply": rope,
        "_CURRENT_GRID_META": {
            "frame_seqlen": 1, "num_new_frames": 2, "h": 1, "w": 1,
        },
        "_CGRAPH_OUTPLACE_KV_ENABLED": outplace,
    }
    module = ast.Module(body=[attention_class, apply_updates], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["CausalWanSelfAttention"], namespace["_apply_cache_updates"]


def _quantization_stub():
    """Identity quantization isolates cache semantics from FP4 rounding."""
    module = types.ModuleType("utils.quant")
    module.quantize_to_fp4 = lambda x, config: x.clone()
    module.copy_quantized_into = lambda dst, src: dst.copy_(src)
    module.k_smooth = lambda k: k - k.mean(dim=-1, keepdim=True)

    def dequantize(slots, max_blocks, num_heads, block_token_size, dtype, device):
        return torch.cat(slots[:max_blocks], dim=0).reshape(
            1, max_blocks * block_token_size, num_heads, 128
        ).to(device=device, dtype=dtype)

    module.dequantize_kv_cache = dequantize
    return module


def _cache(quantized, dtype):
    cache = {
        "quantized": quantized,
        "global_end_index": torch.tensor(0),
        "local_end_index": torch.tensor(0),
    }
    if quantized:
        cache.update({
            "max_blocks": 3,
            "block_token_size": 2,
            "k": [torch.zeros(2, 128, dtype=dtype) for _ in range(3)],
            "v": [torch.zeros(2, 128, dtype=dtype) for _ in range(3)],
        })
    else:
        cache.update({
            "k": torch.zeros(1, 6, 1, 128, dtype=dtype),
            "v": torch.zeros(1, 6, 1, 128, dtype=dtype),
        })
    return cache


class QuantizedKVCacheConsistencyTest(unittest.TestCase):
    @torch.no_grad()
    def test_attention_and_cache_match_unquantized_path(self):
        for outplace, relative_rope, sink, dtype in itertools.product(
            (False, True), (False, True), (0, 2), (torch.float32, torch.bfloat16)
        ):
            with self.subTest(
                outplace=outplace, relative_rope=relative_rope,
                sink=sink, dtype=dtype,
            ):
                attention_class, apply_updates = _load_cache_functions(outplace)
                layer = attention_class(
                    dim=128, num_heads=1, local_attn_size=6,
                    sink_size=sink, qk_norm=False,
                )
                for name in ("q", "k", "v", "o"):
                    setattr(layer, name, nn.Identity())
                layer.max_attention_size = 6
                updater = types.SimpleNamespace(kv_quant_config=None)
                dense_cache = _cache(False, dtype)
                quantized_cache = _cache(True, dtype)
                generator = torch.Generator().manual_seed(104)
                actions = set()

                with patch.dict(sys.modules, {"utils.quant": _quantization_stub()}):
                    # Includes an overwrite during denoising and two cache rolls.
                    for start in (0, 0, 2, 4, 6, 8):
                        x = torch.randn(1, 2, 128, generator=generator) * 0.1
                        x[:, 0] += 0.15 + start * 0.01
                        x[:, 1] += 0.35 + start * 0.01
                        x = x.to(dtype)
                        original = x.clone()
                        kwargs = {
                            "seq_lens": None, "grid_sizes": None,
                            "freqs": None, "block_mask": None,
                            "current_start": start,
                            "use_relative_rope": relative_rope,
                        }
                        expected, dense_update = layer(
                            x, kv_cache=dense_cache, **kwargs
                        )
                        actual, quantized_update = layer(
                            x, kv_cache=quantized_cache, **kwargs
                        )
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertTrue(torch.equal(x, original))
                        actions.add(quantized_update[2]["action"])

                        apply_updates(updater, [dense_cache], [(0, dense_update)])
                        apply_updates(
                            updater, [quantized_cache], [(0, quantized_update)]
                        )
                        for key in ("k", "v"):
                            stored = torch.cat(quantized_cache[key]).reshape(
                                1, 6, 1, 128
                            )
                            self.assertTrue(torch.equal(stored, dense_cache[key]))
                        for key in ("global_end_index", "local_end_index"):
                            self.assertEqual(
                                quantized_cache[key].item(), dense_cache[key].item()
                            )

                self.assertEqual(actions, {"direct_insert", "roll_and_insert"})


if __name__ == "__main__":
    unittest.main()
