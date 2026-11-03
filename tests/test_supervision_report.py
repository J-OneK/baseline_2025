"""Check paired comparisons and refuse inconsistent per-user evidence."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from taac_retrieval.metrics import recommendation_metrics
from taac_retrieval.supervision_report import aggregate, load_prediction


class SupervisionReportTests(unittest.TestCase):
    def test_difference_direction_is_all_minus_click(self):
        # A constant paired improvement has an exact, non-random interval.
        runs = {arm: [] for arm in ['click_infonce', 'all_infonce']}
        predictions = {arm: dict(validation=[], test=[]) for arm in runs}
        for arm, level in [('click_infonce', 0.), ('all_infonce', 1.)]:
            for _ in range(3):
                runs[arm].append({split: {'HR@10': level, 'NDCG@10': level}
                                  for split in ['validation', 'test']})
                for split in ['validation', 'test']:
                    predictions[arm][split].append(dict(hits=np.full(4, level), ndcg=np.full(4, level)))
        comparison = aggregate(runs, predictions)
        for split in ['validation', 'test']:
            for metric in ['HR@10', 'NDCG@10']:
                difference = comparison[split]['differences'][metric]
                self.assertEqual(difference['absolute'], 1.)
                self.assertEqual(difference['paired_seed_and_user_95pct'], [1., 1.])
                self.assertIsNone(difference['relative_to_click'])

    def test_saved_scores_and_user_identity_are_checked(self):
        predictions = np.tile(np.arange(1, 11), (3, 1))
        targets = np.array([1, 3, 99])
        hits, ndcg = recommendation_metrics(predictions, targets)
        expected = {'HR@10': 2 / 3, 'NDCG@10': .5}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.npz'
            np.savez(path, rows=[0, 1, 2], targets=targets, predictions=predictions, hits=hits, ndcg=ndcg)
            saved = load_prediction(path, expected)
            np.testing.assert_array_equal(saved['targets'], targets)
            with self.assertRaises(AssertionError):
                load_prediction(path, {'HR@10': 1., 'NDCG@10': .5})
            np.savez(path, rows=[0, 0, 2], targets=targets, predictions=predictions, hits=hits, ndcg=ndcg)
            with self.assertRaisesRegex(AssertionError, 'Duplicate'):
                load_prediction(path, expected)


if __name__ == '__main__':
    unittest.main()
