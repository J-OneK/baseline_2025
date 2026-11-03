"""Validate supervision semantics, contrastive gradients, and fresh cohorts."""

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from taac_retrieval.cohorts import freeze_cohort
from taac_retrieval.data import PreparedData
from taac_retrieval.objectives import supervision_weights, weighted_bce, sampled_info_nce
from test_retrieval import write_fixture


class ObjectiveTests(unittest.TestCase):
    def test_exposures_clicks_unknowns_are_distinct(self):
        actions = torch.tensor([0, 1, 2, 3])
        torch.testing.assert_close(supervision_weights(actions, 'all_bce'), torch.tensor([0., 1., 1., 0.]))
        torch.testing.assert_close(supervision_weights(actions, 'weighted_bce'), torch.tensor([0., .1, 1., 0.]))
        torch.testing.assert_close(supervision_weights(actions, 'click_bce'), torch.tensor([0., 0., 1., 0.]))
        torch.testing.assert_close(supervision_weights(actions, 'click_infonce'), torch.tensor([0., 0., 1., 0.]))

    def test_info_nce_supervision_scope_controls_exposure_gradients(self):
        actions = torch.tensor([0, 1, 2, 3])
        for objective, expected in [('click_infonce', [False, False, True, False])]:
            with self.subTest(objective=objective):
                positive = torch.zeros(4, requires_grad=True)
                negative = torch.zeros(4, 64, requires_grad=True)
                mask = supervision_weights(actions, objective) > 0
                loss = sampled_info_nce(positive[mask], negative[mask])
                self.assertAlmostEqual(float(loss.detach()), math.log(65), places=5)
                loss.backward()
                self.assertEqual((positive.grad < 0).tolist(), expected)
                self.assertEqual((negative.grad > 0).all(1).tolist(), expected)

    def test_negative_count_does_not_rescale_bce(self):
        expected = torch.tensor(2 * math.log(2))
        for negatives in [1, 64]:
            torch.testing.assert_close(weighted_bce(torch.zeros(3), torch.zeros(3, negatives), torch.ones(3)), expected)

    def test_unlabelled_positions_have_zero_gradient(self):
        positive = torch.tensor([0., -50.], requires_grad=True)
        negative = torch.tensor([[0., 0.], [50., 50.]], requires_grad=True)
        loss = weighted_bce(positive, negative, torch.tensor([1., 0.]))
        self.assertAlmostEqual(float(loss.detach()), 2 * math.log(2), places=6)
        loss.backward()
        self.assertEqual(float(positive.grad[1]), 0.)
        torch.testing.assert_close(negative.grad[1], torch.zeros(2))

    def test_info_nce_has_correct_uniform_value_and_gradient_directions(self):
        positive = torch.zeros(2, requires_grad=True)
        negative = torch.zeros(2, 64, requires_grad=True)
        loss = sampled_info_nce(positive, negative)
        self.assertAlmostEqual(float(loss.detach()), math.log(65), places=5)
        loss.backward()
        self.assertTrue((positive.grad < 0).all())
        self.assertTrue((negative.grad > 0).all())

    def test_invalid_temperature_or_unsupervised_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            sampled_info_nce(torch.zeros(1), torch.zeros(1, 2), temperature=0)
        with self.assertRaises(ValueError):
            weighted_bce(torch.zeros(1), torch.zeros(1, 2), torch.zeros(1))

    def test_frozen_cohort_excludes_previously_tested_users(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['old', 'new']:
                (root / name).mkdir()
                write_fixture(root / name)
            np.save(root / 'old/eval_indices.npy', np.array([0]))
            output = root / 'cohort.npy'
            description = freeze_cohort(root / 'new', root / 'old', output, count=1)
            self.assertEqual(np.load(output).tolist(), [1])
            self.assertEqual(description['overlap_with_previous_test_users'], 0)
            freeze_cohort(root / 'new', root / 'old', output, count=1)
            data = PreparedData(root / 'new')
            with self.assertRaises(ValueError):
                data.set_eval_indices([0, 0])
