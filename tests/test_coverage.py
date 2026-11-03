"""Validate rank budgets, frequency groups and exact candidate filtering."""

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from taac_retrieval.coverage import coverage_summary, diagnose
from taac_retrieval.metrics import exact_topk


class CoverageTests(unittest.TestCase):
    def test_rank_bands_conditional_coverage_and_frequency_boundaries(self):
        predictions = np.tile(np.arange(1, 7), (5, 1))
        targets = np.array([1, 3, 5, 6, 8])
        frequencies = np.zeros(10, dtype=np.int64)
        frequencies[targets] = [0, 1, 5, 6, 51]
        summary, per_user = coverage_summary(predictions, targets, np.arange(5), frequencies, ks=[1, 3, 6])
        np.testing.assert_array_equal(per_user['target_rank'], [1, 3, 5, 6, 0])
        self.assertEqual(summary['all']['Recall@1'], .2)
        self.assertEqual(summary['all']['Recall@3'], .4)
        self.assertEqual(summary['all']['Recall@6'], .8)
        self.assertEqual(summary['all']['conditional_Recall@1_given_Top6'], .25)
        self.assertEqual(summary['all']['target_rank_bands'], {'1_1': 1, '2_3': 1, '4_6': 2, 'outside_6': 1})
        self.assertEqual(summary['tail_1_5']['users'], 2)
        self.assertEqual(summary['mid_6_50']['users'], 1)
        self.assertEqual(summary['head_gt50']['Recall@6'], 0.)
        self.assertAlmostEqual(summary['all']['NDCG@3'], (1 + .5) / 5)

    def test_invalid_candidates_or_budget_are_not_silently_accepted(self):
        counts = np.zeros(10, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, 'never clipped'):
            coverage_summary([[1, 2]], [3], [0], counts, ks=[10])
        with self.assertRaisesRegex(ValueError, 'Duplicate predicted'):
            coverage_summary([[1, 1]], [3], [0], counts, ks=[2])
        with self.assertRaisesRegex(ValueError, 'nonpadding'):
            coverage_summary([[0, 1]], [3], [0], counts, ks=[2])
        with self.assertRaisesRegex(ValueError, 'Duplicate evaluation'):
            coverage_summary([[1, 2], [1, 2]], [3, 3], [0, 0], counts, ks=[2])

    def test_empty_group_is_distinct_from_no_hits(self):
        counts = np.zeros(10, dtype=np.int64)
        summary, _ = coverage_summary([[1, 2, 3]], [4], [0], counts, ks=[1, 3])
        self.assertEqual(summary['cold_0']['Recall@3'], 0.)
        self.assertIsNone(summary['head_gt50']['Recall@3'])
        self.assertIsNone(summary['cold_0']['conditional_Recall@1_given_Top3'])

    def test_streaming_larger_topk_matches_bruteforce_after_history_filter(self):
        generator = torch.Generator().manual_seed(37)
        queries = torch.randn(3, 8, generator=generator)
        vectors = torch.randn(41, 8, generator=generator)
        seen = torch.tensor([[1, 7, 0], [4, 8, 0], [10, 12, 0]])
        score = queries @ vectors.T
        score[:, 0] = -torch.inf
        score.scatter_(1, seen, -torch.inf)
        expected = score.topk(20).indices
        actual = exact_topk(queries, vectors, seen, k=20, chunk_size=7)
        torch.testing.assert_close(actual, expected)

    def test_diagnostic_refuses_any_existing_output_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'coverage'
            output.with_suffix('.npz').write_bytes(b'kept')
            with self.assertRaises(FileExistsError):
                diagnose(Namespace(output=str(output), checkpoint='missing', cache='missing'))
            self.assertEqual(output.with_suffix('.npz').read_bytes(), b'kept')


if __name__ == '__main__':
    unittest.main()
