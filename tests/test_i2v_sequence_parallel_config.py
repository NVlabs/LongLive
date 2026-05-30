import unittest
from types import SimpleNamespace

from wan_5b.distributed.sp_training import (
    sp_training_sequence_frame_count,
    validate_sequence_parallel_training_config,
)


class I2VSequenceParallelConfigTest(unittest.TestCase):
    def test_i2v_training_frames_use_full_configured_sequence(self):
        cfg = SimpleNamespace(
            i2v=True,
            independent_first_frame=True,
            image_or_video_shape=[1, 96, 48, 44, 80],
        )

        self.assertEqual(sp_training_sequence_frame_count(cfg), 96)

    def test_i2v_sequence_parallel_partitions_the_96_frame_sequence(self):
        cfg = SimpleNamespace(
            i2v=True,
            independent_first_frame=True,
            image_or_video_shape=[1, 96, 48, 44, 80],
        )

        validate_sequence_parallel_training_config(
            cfg,
            sp_size=4,
            num_frame_per_block=8,
        )

    def test_i2v_sequence_parallel_rejects_non_block_aligned_length(self):
        cfg = SimpleNamespace(
            i2v=True,
            independent_first_frame=True,
            image_or_video_shape=[1, 97, 48, 44, 80],
        )

        with self.assertRaisesRegex(ValueError, r"training latent frames"):
            validate_sequence_parallel_training_config(
                cfg,
                sp_size=4,
                num_frame_per_block=8,
            )

    def test_t2v_sequence_parallel_uses_full_sequence(self):
        cfg = SimpleNamespace(
            i2v=False,
            independent_first_frame=False,
            image_or_video_shape=[1, 96, 48, 44, 80],
        )

        self.assertEqual(sp_training_sequence_frame_count(cfg), 96)
        validate_sequence_parallel_training_config(
            cfg,
            sp_size=4,
            num_frame_per_block=8,
        )


if __name__ == "__main__":
    unittest.main()
