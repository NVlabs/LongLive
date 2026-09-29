import unittest

from utils.sampler import build_training_sampler


class DistributedSamplerSeedTest(unittest.TestCase):
    def test_shared_seed_partitions_dataset_without_overlap(self):
        dataset = list(range(24))
        samplers = [
            build_training_sampler(
                dataset,
                seed=1234,
                rank=rank,
                num_replicas=4,
            )
            for rank in range(4)
        ]
        for sampler in samplers:
            sampler.set_epoch(3)

        rank_indices = [list(sampler) for sampler in samplers]
        combined_indices = [index for indices in rank_indices for index in indices]

        self.assertEqual(len(combined_indices), len(dataset))
        self.assertEqual(len(set(combined_indices)), len(dataset))
        self.assertEqual(set(combined_indices), set(range(len(dataset))))

    def test_explicit_rank_requires_replica_count(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            build_training_sampler(list(range(4)), seed=1234, rank=0)


if __name__ == "__main__":
    unittest.main()
