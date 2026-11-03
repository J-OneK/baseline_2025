"""Check fixed candidate budgets, protected Top-10 and development safeguards."""

import unittest

import numpy as np

from taac_retrieval.candidate_fusion import aligned_inputs, candidate_union, quota_candidates, select_quota


class CandidateFusionTests(unittest.TestCase):
    def test_quota_keeps_prefix_and_deduplicates_padding_before_refill(self):
        baseline = np.arange(1, 13)[None]
        extra = np.array([[0, -1, 1, 1, 20, 20, 21, 22]])
        found, added = quota_candidates(baseline, extra, quota=3, budget=12, protected=3)
        np.testing.assert_array_equal(found, [[1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22]])
        self.assertEqual(added.tolist(), [3])
        found, added = quota_candidates(baseline, np.zeros((1, 5), dtype=int), quota=3, budget=12, protected=3)
        np.testing.assert_array_equal(found, baseline)
        self.assertEqual(added.tolist(), [0])

    def test_zero_quota_is_exact_baseline_and_top10_is_protected(self):
        baseline = np.arange(1, 501)[None]
        extra = np.arange(501, 1001)[None]
        zero, added = quota_candidates(baseline, extra, quota=0)
        np.testing.assert_array_equal(zero, baseline)
        self.assertEqual(added.tolist(), [0])
        fused, _ = quota_candidates(baseline, extra, quota=250)
        np.testing.assert_array_equal(fused[:, :10], baseline[:, :10])
        self.assertEqual(len(np.unique(fused)), 500)
        with self.assertRaises(ValueError):
            quota_candidates(baseline, extra, quota=491)

    def test_selection_enforces_all_groups_and_retains_zero_on_tie(self):
        def row(quota, overall, cold, tail):
            return dict(quota=quota, groups={name: {'Recall@500': value}
                         for name, value in [('all', overall), ('cold_0', cold), ('tail_1_5', tail)]})
        rows = [row(0, .2, .1, .02), row(50, .195, .11, .03),
                row(100, .189, .3, .3), row(250, .195, .15, .01)]
        selection = select_quota(rows)
        self.assertEqual(selection['selected_quota'], 50)
        self.assertFalse(selection['rows'][2]['eligible'])
        self.assertFalse(selection['rows'][3]['eligible'])
        selection = select_quota([row(0, .2, .1, .02), row(50, .2, .1, .02)])
        self.assertEqual(selection['selected_quota'], 0)

    def test_input_identity_and_validation_only_metadata_are_required(self):
        baseline = dict(predictions=np.array([[1, 2, 3]]), rows=np.array([0]), targets=np.array([6]), seen=np.array([[4, 0]]))
        extra = {key: value.copy() for key, value in baseline.items()}
        extra['predictions'] = np.array([[0, 3, 7]])
        meta = dict(split='val', test_targets_read=False, evaluation_cohort_sha256='same', cache_signature={'seed': 1})
        aligned_inputs(baseline, extra, meta, meta, num_items=10)
        extra['rows'] = np.array([1])
        with self.assertRaisesRegex(ValueError, 'rows'):
            aligned_inputs(baseline, extra, meta, meta, 10)
        extra['rows'] = np.array([0])
        with self.assertRaisesRegex(ValueError, 'validation-only'):
            aligned_inputs(baseline, extra, meta, dict(meta, split='test'), 10)

    def test_union_is_labeled_with_larger_budget_and_extra_hits(self):
        baseline = np.array([[1, 2, 3], [4, 5, 6]])
        extra = np.array([[2, 7, 0], [6, 8, 9]])
        targets = np.array([7, 5])
        counts = np.zeros(10, dtype=int)
        counts[5] = 60
        result = candidate_union(baseline, extra, targets, counts, k=3)
        self.assertEqual(result['maximum_union_budget'], 6)
        self.assertFalse(result['is_fixed_500_comparison'])
        self.assertEqual(result['groups']['all']['union_candidate_coverage'], 1.)
        self.assertEqual(result['groups']['all']['additional_hits'], 1)


if __name__ == '__main__':
    unittest.main()
