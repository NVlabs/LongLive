# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: The LongLive contributors
# SPDX-License-Identifier: Apache-2.0
# Provenance: Copied from the source snapshot listed below.
# Source: https://github.com/NVlabs/LongLive @ 0308b126accba9440b8caa45bcf7bec0877933e1 :: tests/test_i2v_dataset_frame_accounting.py
# Source: https://github.com/Aaronhuang-778/Long-WAM @ 87b87ceb65582c46744792a77f2f61754f8274f0 :: longlive/third_party/LongLive/tests/test_i2v_dataset_frame_accounting.py
# License texts and component notes: see THIRD_PARTY_NOTICES.md at the repository root.
# End Long-WAM attribution.

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
