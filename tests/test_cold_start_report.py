"""Guard cold/tail improvements against overall or per-bucket regressions."""

import unittest

from taac_retrieval.cold_start_report import select_configuration


def arm(name, probability, ndcg=.1, cold=.2, tail=.1):
    return dict(name=name, probability=probability,
                groups=dict(all={'NDCG@10': {'mean': ndcg}},
                            cold_0={'Recall@500': {'mean': cold}},
                            tail_1_5={'Recall@500': {'mean': tail}}))


class ColdStartSelectionTests(unittest.TestCase):
    def setUp(self):
        self.settings = dict(minimum_overall_ndcg_retention=.95, require_cold_and_tail_each_not_worse=True)

    def test_overall_quality_gate_rejects_larger_rare_macro(self):
        result = select_configuration([arm('click', 0), arm('mask', .3, ndcg=.094, cold=.5, tail=.3)], self.settings)
        self.assertEqual(result['selected_arm'], 'click')
        self.assertIn('overall_ndcg_below_retention_gate', result['rows'][1]['rejection_reasons'])

    def test_macro_cannot_hide_one_rare_bucket_regression(self):
        result = select_configuration([arm('click', 0), arm('mask', .3, cold=.5, tail=.09)], self.settings)
        self.assertTrue(result['baseline_retained'])
        self.assertIn('tail_recall500_decreased', result['rows'][1]['rejection_reasons'])

    def test_strict_improvement_and_smaller_probability_tie_break(self):
        result = select_configuration([arm('click', 0), arm('large', .7, cold=.3),
                                       arm('small', .3, cold=.3)], self.settings)
        self.assertEqual(result['selected_arm'], 'small')
        same = select_configuration([arm('click', 0), arm('same', .3)], self.settings)
        self.assertEqual(same['selected_arm'], 'click')

    def test_exact_retention_boundary_is_eligible(self):
        result = select_configuration([arm('click', 0), arm('mask', .3, ndcg=.095, tail=.11)], self.settings)
        self.assertEqual(result['selected_arm'], 'mask')

    def test_invalid_metrics_and_ambiguous_control_are_rejected(self):
        with self.assertRaises(ValueError):
            select_configuration([arm('click', 0), arm('bad', .3, cold=float('nan'))], self.settings)
        with self.assertRaises(ValueError):
            select_configuration([arm('first', 0), arm('second', 0)], self.settings)


if __name__ == '__main__':
    unittest.main()
