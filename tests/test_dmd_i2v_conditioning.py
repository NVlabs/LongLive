import unittest

import torch

from utils.i2v_conditioning import (
    _get_i2v_context_frames,
    _i2v_loss_mask_like,
    _overwrite_i2v_context,
)


class DMDI2VConditioningTest(unittest.TestCase):
    def test_overwrite_i2v_context_keeps_only_initial_frames_clean(self):
        tensor = torch.randn(2, 4, 3, 2, 2)
        original = tensor.clone()
        initial_latent = torch.full((2, 1, 3, 2, 2), 7.0)

        context_frames = _get_i2v_context_frames(tensor, initial_latent)
        conditioned = _overwrite_i2v_context(tensor, initial_latent, context_frames)

        self.assertEqual(context_frames, 1)
        self.assertTrue(torch.equal(conditioned[:, :1], initial_latent))
        self.assertTrue(torch.equal(conditioned[:, 1:], original[:, 1:]))

    def test_i2v_loss_mask_excludes_context_frames(self):
        tensor = torch.randn(2, 4, 3, 2, 2)
        mask = _i2v_loss_mask_like(tensor, context_frames=1)

        self.assertFalse(mask[:, :1].any())
        self.assertTrue(mask[:, 1:].all())

    def test_overwrite_i2v_context_keeps_chunk_length_unchanged(self):
        initial_latent = torch.full((2, 1, 3, 2, 2), 5.0)
        noise_chunk = torch.randn(2, 4, 3, 2, 2)

        first_chunk = _overwrite_i2v_context(noise_chunk, initial_latent, context_frames=1)

        self.assertEqual(first_chunk.shape[1], 4)
        self.assertTrue(torch.equal(first_chunk[:, :1], initial_latent))
        self.assertTrue(torch.equal(first_chunk[:, 1:], noise_chunk[:, 1:]))

    def test_i2v_context_must_not_cover_entire_clip(self):
        tensor = torch.randn(2, 1, 3, 2, 2)
        initial_latent = torch.randn(2, 1, 3, 2, 2)

        with self.assertRaises(ValueError):
            _get_i2v_context_frames(tensor, initial_latent)


if __name__ == "__main__":
    unittest.main()
