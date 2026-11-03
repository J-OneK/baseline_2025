"""Protect validation-only selection and seed-aware uncertainty reporting."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from taac_retrieval.study import DEVELOPMENT_VARIANTS, lock_selection
from taac_retrieval.week_report import confidence_intervals


class StudyTests(unittest.TestCase):
    def test_seed_uncertainty_is_not_hidden_by_user_bootstrap(self):
        differences = np.array([[1.] * 30, [-1.] * 30, [1.] * 30])
        result = confidence_intervals(differences, samples=1000)
        np.testing.assert_allclose(result['paired_user_95pct'], [1/3, 1/3])
        self.assertLess(result['paired_seed_and_user_95pct'][0], 0)
        self.assertGreater(result['paired_seed_and_user_95pct'][1], 0)

    def test_selection_uses_validation_and_cannot_be_silently_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, study, reports = root / 'old', root / 'study', root / 'reports'
            config = dict(development_seeds=[1, 2], selection='mean validation NDCG')
            settings = dict(epochs=2, batch_size=4, hidden=8, blocks=1, heads=1, dropout=.2, lr=.001)
            for name in ['baseline'] + list(DEVELOPMENT_VARIANTS):
                for seed in [1, 2]:
                    folder = baseline / f'baseline_seed{seed}' if name == 'baseline' else study / 'development' / f'{name}_seed{seed}'
                    folder.mkdir(parents=True)
                    score = .2 if name == 'click_infonce64' else .1
                    history = [dict(epoch=1, validation={'NDCG@10': score, 'HR@10': .3}),
                               dict(epoch=2, validation={'NDCG@10': score-.01, 'HR@10': .4})]
                    (folder / 'history.json').write_text(json.dumps(history))
                    (folder / 'config.json').write_text(json.dumps(dict(settings, shared_initialization_sha256=str(seed))))
                    # It is deliberately invalid: selection must never read test/result JSON.
                    (folder / 'result.json').write_text('unopened test result')
            selected = lock_selection(study, baseline, config, reports)
            self.assertEqual(selected['selected_variant'], 'click_infonce64')
            self.assertFalse(selected['test_used_for_selection'])
            lock_selection(study, baseline, config, reports)
            path = study / 'development/weighted_bce_seed1/history.json'
            changed = json.loads(path.read_text())
            changed[0]['validation']['NDCG@10'] = 1.
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                lock_selection(study, baseline, config, reports)
