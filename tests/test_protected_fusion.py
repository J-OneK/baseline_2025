"""Check rare-candidate protection, replacement ordering and fixed budgets."""

import unittest

import numpy as np

from taac_retrieval.protected_fusion import protected_candidates


class ProtectedFusionTests(unittest.TestCase):
    def test_preserves_all_old_rare_candidates_and_top_prefix(self):
        baseline = np.arange(1, 13)[None]
        counts = np.full(31, 60)
        counts[[4, 11, 12, 20, 21, 22]] = [0, 1, 5, 0, 2, 5]
        extra = np.array([[0, 4, 20, 20, 23, 21, 22, -1]])
        found, replaced = protected_candidates(baseline, extra, counts, quota=3, budget=12, protected=3)
        np.testing.assert_array_equal(found, [[1, 2, 3, 4, 5, 6, 7, 11, 12, 20, 21, 22]])
        # Budget is 12: last removable IDs 8,9,10 are removed; 7 remains.
        self.assertEqual(found.shape[1], 12)
        self.assertEqual(replaced.tolist(), [3])
        self.assertTrue({4, 11, 12}.issubset(set(found[0])))

    def test_replacement_count_is_capped_by_available_removable_items(self):
        baseline = np.arange(1, 7)[None]
        counts = np.zeros(20, dtype=int)
        counts[5] = 6
        extra = np.array([[10, 11, 12]])
        found, replaced = protected_candidates(baseline, extra, counts, quota=3, budget=6, protected=2)
        np.testing.assert_array_equal(found, [[1, 2, 3, 4, 6, 10]])
        self.assertEqual(replaced.tolist(), [1])

    def test_only_new_rare_extra_items_are_admitted_and_zero_quota_is_identity(self):
        baseline = np.arange(1, 7)[None]
        counts = np.full(20, 6)
        counts[[8, 9]] = [5, 0]
        extra = np.array([[0, -1, 1, 7, 8, 8, 9]])
        found, replaced = protected_candidates(baseline, extra, counts, quota=3, budget=6, protected=2)
        np.testing.assert_array_equal(found, [[1, 2, 3, 4, 8, 9]])
        self.assertEqual(replaced.tolist(), [2])
        control, replaced = protected_candidates(baseline, extra, counts, quota=0, budget=6, protected=2)
        np.testing.assert_array_equal(control, baseline)
        self.assertEqual(replaced.tolist(), [0])

    def test_all_rare_baseline_or_no_extra_keeps_every_candidate(self):
        baseline = np.arange(1, 7)[None]
        counts = np.zeros(20, dtype=int)
        found, replaced = protected_candidates(baseline, np.array([[10, 11, 12]]), counts, quota=3, budget=6, protected=2)
        np.testing.assert_array_equal(found, baseline)
        self.assertEqual(replaced.tolist(), [0])
        counts[:] = 6
        found, replaced = protected_candidates(baseline, np.zeros((1, 4), dtype=int), counts, quota=3, budget=6, protected=2)
        np.testing.assert_array_equal(found, baseline)

    def test_invalid_catalog_or_incomplete_baseline_is_rejected(self):
        counts = np.zeros(10, dtype=int)
        with self.assertRaises(ValueError):
            protected_candidates([[1, 1, 2]], [[3]], counts, quota=1, budget=3, protected=1)
        with self.assertRaises(ValueError):
            protected_candidates([[1, 2, 3]], [[10]], counts, quota=1, budget=3, protected=1)


if __name__ == '__main__':
    unittest.main()
