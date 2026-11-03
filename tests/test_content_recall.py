"""Validate history-only text aggregation, catalog exclusion and denominators."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from taac_retrieval.content_recall import prepare_candidates, content_queries, summarize, union_summary
from taac_retrieval.metrics import exact_topk


class ContentRecallTests(unittest.TestCase):
    def test_click_mean_precedes_exposure_and_uses_raw_magnitudes(self):
        raw = torch.tensor([[0., 0.], [1., 0.], [0., 3.], [9., 0.]])
        eligible = torch.tensor([False, True, True, True])
        items = torch.tensor([[1, 2, 3], [3, 0, 0], [1, 2, 0]])
        actions = torch.tensor([[2, 2, 1], [1, 0, 0], [3, 3, 0]])
        query, valid, source = content_queries(raw, eligible, items, actions)
        torch.testing.assert_close(query[0], torch.tensor([1., 3.]) / np.sqrt(10))
        torch.testing.assert_close(query[1], torch.tensor([1., 0.]))
        torch.testing.assert_close(query[2], torch.zeros(2))
        self.assertEqual(valid.tolist(), [True, True, False])
        self.assertEqual(source.tolist(), [1, 2, 0])

    def test_bad_content_is_excluded_and_history_is_filtered(self):
        raw = torch.tensor([[100., 0.], [1., 0.], [0., 1.], [-1., 0.], [float('nan'), 1.], [0., 0.]])
        eligible, vectors, ids, mapping = prepare_candidates(raw, torch.ones(6, dtype=torch.bool))
        self.assertEqual(ids.tolist(), [0, 1, 2, 3])
        self.assertEqual(eligible.tolist(), [False, True, True, True, False, False])
        query, valid, _ = content_queries(raw, eligible, torch.tensor([[1, 4]]), torch.tensor([[2, 2]]))
        self.assertTrue(valid.item())
        selected = exact_topk(query, vectors, mapping[torch.tensor([[1, 4]])], k=2, chunk_size=2)
        self.assertEqual(ids[selected].tolist(), [[2, 3]])

    def test_empty_query_counts_as_a_miss_not_an_excluded_user(self):
        predictions = np.array([[2, 3], [0, 0], [4, 3]])
        groups, per_user = summarize(predictions, np.array([2, 1, 4]), np.array([0, 1, 2]),
                                      np.array([0, 0, 0, 2, 8]), np.array([True, False, True]),
                                      np.array([True, True, True]), [1, 2])
        self.assertEqual(groups['all']['users'], 3)
        self.assertAlmostEqual(groups['all']['Recall@1'], 2/3)
        self.assertEqual(groups['cold_0']['Recall@1'], .5)
        self.assertEqual(per_user['target_rank'].tolist(), [1, 0, 1])
        with self.assertRaises(ValueError):
            summarize(predictions, np.array([2, 0, 4]), np.array([0, 1, 2]), np.zeros(5),
                      np.array([True, False, True]), np.ones(3, dtype=bool), [1, 2])

    def test_union_budget_and_identity_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'learned.npz'
            np.savez(path, rows=[0, 1], targets=[2, 4], predictions=[[2, 1], [3, 1]])
            union = union_summary(path, np.array([[1, 3], [4, 2]]), np.array([2, 4]), np.array([0, 1]),
                                  np.array([0, 8, 0, 2, 1]), 2)
            self.assertEqual(union['maximum_union_candidates'], 4)
            self.assertEqual(union['groups']['all']['union_hits'], 2)
            self.assertEqual(union['groups']['tail_1_5']['additional_hits_from_content'], 1)
            with self.assertRaises(AssertionError):
                union_summary(path, np.array([[1, 3], [4, 2]]), np.array([2, 4]), np.array([1, 0]),
                              np.zeros(5), 2)


if __name__ == '__main__':
    unittest.main()
