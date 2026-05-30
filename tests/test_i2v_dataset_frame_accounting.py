import tempfile
import unittest
from pathlib import Path

from utils.dataset import MultiVideoConcatDataset


class I2VDatasetFrameAccountingTest(unittest.TestCase):
    def _dataset(self, root, latent_frames):
        (root / "video" / "sample").mkdir(parents=True)
        (root / "caption" / "sample").mkdir(parents=True)
        total_raw_frames = 1 + (latent_frames - 1) * 4
        return MultiVideoConcatDataset(
            data_dir=str(root),
            video_size=(704, 1280),
            total_frames=total_raw_frames,
            independent_first_frame=True,
            num_frame_per_block=8,
            temporal_compression_ratio=4,
        )

    def test_96_frame_i2v_uses_regular_8_frame_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(Path(tmp), latent_frames=96)

            self.assertEqual(dataset.first_chunk_latent_frames, 8)
            self.assertEqual(dataset.first_chunk_frames, 29)
            self.assertEqual(dataset.total_segments, 12)

    def test_33_frame_i2v_keeps_legacy_one_plus_block_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(Path(tmp), latent_frames=33)

            self.assertEqual(dataset.first_chunk_latent_frames, 9)
            self.assertEqual(dataset.first_chunk_frames, 33)
            self.assertEqual(dataset.total_segments, 4)


if __name__ == "__main__":
    unittest.main()
