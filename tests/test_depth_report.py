"""Selection must follow declared validation metrics, with nested shared init."""

import unittest

from taac_retrieval.depth_report import outcome_summary, select_depth, verify_nested_initializations


def arm(depth, ndcg=.1, hr=.2):
    return dict(depth=depth, name=f'depth_{depth}', groups={'all': {
        'NDCG@10': {'mean': ndcg}, 'Recall@10': {'mean': hr}}})


class DepthReportTests(unittest.TestCase):
    def test_primary_ndcg_precedes_hr_or_depth(self):
        result = select_depth([arm(1, .1, .9), arm(2, .11, .12), arm(4, .105, 1.)])
        self.assertEqual(result['selected_depth'], 2)

    def test_hr_breaks_ndcg_tie_then_smaller_depth(self):
        result = select_depth([arm(1), arm(2, hr=.3), arm(4, hr=.3)])
        self.assertEqual(result['selected_depth'], 2)
        self.assertTrue(select_depth([arm(1), arm(2), arm(4)])['baseline_retained'])

    def test_duplicate_missing_control_and_nonfinite_are_rejected(self):
        for rows in [[], [arm(2)], [arm(1), arm(1)], [arm(1), arm(2, ndcg=float('nan'))],
                     [arm(1), arm(4, hr=-.1)], [arm(1), arm(1.5)], [arm(1), arm(2, ndcg=.3, hr=.2)]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                select_depth(rows)

    def test_nested_initialization_includes_added_second_layer(self):
        first = {'item_emb.weight':'a', 'attention.0.q_linear.weight':'b', 'final_norm.weight':'c'}
        second = dict(first, **{'attention.1.q_linear.weight':'d'})
        fourth = dict(second, **{'attention.2.q_linear.weight':'e', 'attention.3.q_linear.weight':'f'})
        result = verify_nested_initializations({1:first, 2:second, 4:fourth})
        self.assertEqual([row['matching_parameters'] for row in result], [3, 4])
        fourth['attention.1.q_linear.weight']='wrong'
        with self.assertRaisesRegex(AssertionError, 'Unmatched initial'):
            verify_nested_initializations({1:first, 2:second, 4:fourth})

    def test_missing_shared_parameters_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, 'missing a shared'):
            verify_nested_initializations({1:{'item_emb.weight':'a'}, 2:{'attention.1.weight':'b'}})
        with self.assertRaises(AssertionError):
            verify_nested_initializations({2:{'item_emb.weight':'a'}})

    def test_interpretation_uses_selected_arm_and_fixed_epoch_evidence(self):
        first, second = arm(1, .1, .2), arm(2, .11, .21)
        first.update(runs=[{'training_seconds':100}], groups={**first['groups'], 'tail_1_5': {'Recall@500': .1}})
        second.update(runs=[{'training_seconds':110}], groups={**second['groups'], 'tail_1_5': {'Recall@500': .09}})
        result = dict(arms={'depth_1':first, 'depth_2':second}, model_selection={'selected_arm':'depth_2'},
                      paired_differences={'depth_2':{'all':{'NDCG@10':{'paired_seed_and_user_95pct':[-.01,.02]}}}},
                      fixed_epoch5_validation={'depth_1':[{'NDCG@10':.1,'HR@10':.2}],
                                               'depth_2':[{'NDCG@10':.09,'HR@10':.19}]})
        summary = outcome_summary(result)
        self.assertEqual(summary['selected_depth'], 2)
        self.assertTrue(summary['ndcg_interval_crosses_zero'])
        self.assertTrue(summary['best_mean_higher'])
        self.assertTrue(summary['selected_fixed_epoch5_below_control'])
        self.assertFalse(summary['tail_recall500_improved'])
        self.assertAlmostEqual(summary['training_time_relative'], .1)
        result['model_selection']['selected_arm'] = 'depth_1'
        summary = outcome_summary(result)
        self.assertIsNone(summary['ndcg_interval_crosses_zero'])
        self.assertFalse(summary['selected_fixed_epoch5_below_control'])


if __name__ == '__main__':
    unittest.main()
