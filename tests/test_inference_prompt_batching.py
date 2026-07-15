import unittest

import torch

from utils.prompt_conditioning import encode_prompt_blocks


class _RecordingTextEncoder:
    def __init__(self):
        self.text_prompts = None

    def __call__(self, *, text_prompts):
        self.text_prompts = text_prompts
        prompt_embeds = torch.arange(len(text_prompts), dtype=torch.float32)
        return {"prompt_embeds": prompt_embeds.reshape(-1, 1, 1)}


class InferencePromptBatchingTest(unittest.TestCase):
    def test_prompt_blocks_are_batched_per_video(self):
        text_encoder = _RecordingTextEncoder()
        text_prompts = [
            ["sample-0-block-0", "sample-0-block-1"],
            ["sample-1-block-0", "sample-1-block-1"],
        ]

        conditional_dict, conditional_dict_list = encode_prompt_blocks(
            text_encoder, text_prompts, batch_size=2
        )

        self.assertEqual(
            text_encoder.text_prompts,
            [
                "sample-0-block-0",
                "sample-0-block-1",
                "sample-1-block-0",
                "sample-1-block-1",
            ],
        )
        self.assertEqual(conditional_dict["prompt_embeds"].shape[0], 4)
        self.assertEqual(len(conditional_dict_list), 2)
        self.assertTrue(
            torch.equal(
                conditional_dict_list[0]["prompt_embeds"].flatten(),
                torch.tensor([0.0, 2.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                conditional_dict_list[1]["prompt_embeds"].flatten(),
                torch.tensor([1.0, 3.0]),
            )
        )

    def test_prompt_batch_size_must_match_noise(self):
        with self.assertRaisesRegex(ValueError, "2 samples"):
            encode_prompt_blocks(
                _RecordingTextEncoder(), [["only-one-sample"]], batch_size=2
            )

    def test_samples_must_have_equal_block_counts(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            encode_prompt_blocks(
                _RecordingTextEncoder(), [["a", "b"], ["c"]], batch_size=2
            )


if __name__ == "__main__":
    unittest.main()
