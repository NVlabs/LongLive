# Adopted from https://github.com/Wan-Video/Wan2.2
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.cuda.amp as amp

from ..modules.model import sinusoidal_embedding_1d
from .ulysses import distributed_attention, distributed_flex_attention
from .util import gather_forward, get_rank, get_world_size
import math


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


from utils.position_embedding_utils import (
    compute_temporal_freqs as _compute_temporal_freqs,
    select_temporal_offset_for_sample,
)


@torch.amp.autocast('cuda', enabled=False)
def sp_rope_apply(
    x,
    grid_sizes,
    freqs,
    t_scale=1.0,
    method="linear",
    original_seq_len=None,
    temporal_offset=0.0,
):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    n, c = x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        local_f = f
        sp_rank = get_rank()
        start_frame = sp_rank * local_f
        seq_len = local_f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        temporal_offset_i = select_temporal_offset_for_sample(
            temporal_offset, i, local_f, start_frame=start_frame)
        temporal_freqs = _compute_temporal_freqs(
            freqs[0], local_f, start_frame, t_scale, x.device,
            method=method, original_seq_len=original_seq_len,
            temporal_offset=temporal_offset_i)
        freqs_i = torch.cat([
            temporal_freqs.view(local_f, 1, 1, -1).expand(local_f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(local_f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(local_f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]
    e = torch.chunk(e, get_world_size(), dim=1)[get_rank()]
    e0 = torch.chunk(e0, get_world_size(), dim=1)[get_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = gather_forward(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16,
                    t_scale=1.0, method="linear", original_seq_len=None,
                    temporal_offset=0.0):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = sp_rope_apply(q, grid_sizes, freqs, t_scale=t_scale,
                      method=method, original_seq_len=original_seq_len,
                      temporal_offset=temporal_offset)
    k = sp_rope_apply(k, grid_sizes, freqs, t_scale=t_scale,
                      method=method, original_seq_len=original_seq_len,
                      temporal_offset=temporal_offset)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x




def sp_dit_causal_forward_train(
    self,
    x,
    t,
    context,
    seq_len,
    clean_x=None,
    aug_t=None,
    clip_fea=None,
    y=None,
):
    r"""
    Forward pass through the diffusion model

    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    if self.model_type == 'i2v':
        assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    # Construct the blockwise causal attention mask. Frames are sharded across
    # SP ranks, so total frames = local frames per rank * sp_size.
    sp_size = get_world_size()
    # Recreate mask when batch size changes to avoid Triton broadcasting bug
    current_batch_size = x.shape[0]
    if self.block_mask is None or self._block_mask_batch_size != current_batch_size:
        self._block_mask_batch_size = current_batch_size
        if clean_x is not None:
            if self.independent_first_frame:
                raise NotImplementedError()
            else:
                self.block_mask = self._prepare_teacher_forcing_mask_natural(
                    device, num_frames=x.shape[2] * sp_size,
                    frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                    num_frame_per_block=self.num_frame_per_block,
                    sp_size=sp_size,
                    batch_size=current_batch_size,
                )
        else:
            if self.independent_first_frame:
                self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                    device, num_frames=x.shape[2] * sp_size,
                    frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                    num_frame_per_block=self.num_frame_per_block,
                    batch_size=current_batch_size,
                )
            else:
                self.block_mask = self._prepare_blockwise_causal_attn_mask(
                    device, num_frames=x.shape[2] * sp_size,
                    frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                    num_frame_per_block=self.num_frame_per_block,
                    batch_size=current_batch_size,
                )

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]

    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    max_len = int(seq_lens.max().item())
    assert max_len > 0, "Token sequence length is zero after patch embedding"
    # Pad all samples to the batch max length instead of the first sample length
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, max_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        raise NotImplementedError(f"t.shape should be [B, F], but got {t.shape}")
    bt = t.size(0)
    t_len = t.size(1)
    t_ori_shape = t.shape
    t = t.flatten()
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, t_len)).type_as(x))
    e0 = self.time_projection(e).unflatten(2, (6, self.dim)) # B, F, 6, C

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clean_x is not None:
        clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
        clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

        seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
        clean_x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
        ])

        x = torch.cat([clean_x, x], dim=1)
        if aug_t is None:
            aug_t = torch.zeros(t_ori_shape, device=t.device, dtype=t.dtype)
        bt_clean = aug_t.size(0)
        t_clean_len = aug_t.size(1)
        aug_t = aug_t.flatten()
        e_clean = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, aug_t).unflatten(0, (bt_clean, t_clean_len)).type_as(x))
        e0_clean = self.time_projection(e_clean).unflatten(2, (6, self.dim))
        e0 = torch.cat([e0_clean, e0], dim=1)


    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
        block_mask=self.block_mask,
        t_scale=self.t_scale,
        use_relative_rope=getattr(self, "use_relative_rope", False),
        method=getattr(self, "rope_method", "linear"),
        original_seq_len=getattr(self, "original_seq_len", None),
        temporal_offset=getattr(self, "rope_temporal_offset", 0.0),
    )

    def create_custom_forward(module):
        def custom_forward(*inputs, **kwargs):
            return module(*inputs, **kwargs)
        return custom_forward

    for block in self.blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, **kwargs,
                use_reentrant=False,
            )
        else:
            x = block(x, **kwargs)

    if clean_x is not None:
        x = x[:, x.shape[1] // 2:]

    # head
    x = self.head(x, e.unsqueeze(2))

    x = self.unpatchify(x, grid_sizes)
    return torch.stack(x)


def sp_causal_attn_forward(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    block_mask,
    kv_cache=None,
    current_start=0,
    cache_start=None,
    t_scale=1.0,
    use_relative_rope=False,
    method="linear",
    original_seq_len=None,
    temporal_offset=0.0,
    **kwargs,
):
    r"""
    Args:
        x(Tensor): Shape [B, L, num_heads, C / num_heads]
        seq_lens(Tensor): Shape [B]
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        block_mask (BlockMask)
        t_scale (float): Temporal RoPE interpolation scale. <1.0 compresses positions.
        method (str): RoPE method. This release supports "linear".
        original_seq_len (int): Unused by the release linear RoPE path.
    """

    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    if cache_start is None:
        cache_start = current_start

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    if kv_cache is None:
        # Teacher-forcing training doubles sequence length with clean/noisy halves.
        is_tf = (s == seq_lens[0].item() * 2)
        if is_tf:
            q_chunk = torch.chunk(q, 2, dim=1)
            k_chunk = torch.chunk(k, 2, dim=1)
            roped_query = []
            roped_key = []
            # rope should be same for clean and noisy parts
            for ii in range(2):
                rq = sp_rope_apply(q_chunk[ii], grid_sizes, freqs, t_scale=t_scale,
                                   method=method, original_seq_len=original_seq_len,
                                   temporal_offset=temporal_offset).type_as(v)
                rk = sp_rope_apply(k_chunk[ii], grid_sizes, freqs, t_scale=t_scale,
                                   method=method, original_seq_len=original_seq_len,
                                   temporal_offset=temporal_offset).type_as(v)
                roped_query.append(rq)
                roped_key.append(rk)

            roped_query = torch.cat(roped_query, dim=1)
            roped_key = torch.cat(roped_key, dim=1)

            x = distributed_flex_attention(
                roped_query,
                roped_key,
                v,
                block_mask,
            )

        else:
            roped_query = sp_rope_apply(q, grid_sizes, freqs, t_scale=t_scale,
                                        method=method, original_seq_len=original_seq_len,
                                        temporal_offset=temporal_offset).type_as(v)
            roped_key = sp_rope_apply(k, grid_sizes, freqs, t_scale=t_scale,
                                      method=method, original_seq_len=original_seq_len,
                                      temporal_offset=temporal_offset).type_as(v)

            x = distributed_flex_attention(
                roped_query,
                roped_key,
                v,
                block_mask,
            )

    else:
        raise NotImplementedError()

    # output
    x = x.flatten(2)
    x = self.o(x)
    
    # Return both output and cache update info
    if kv_cache is not None:
        raise NotImplementedError()
    else:
        return x
