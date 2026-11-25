import unittest

import numpy as np
import torch

from taac_retrieval.channel_features import channel_features, normalize_to_original
from taac_retrieval.channel_ranker import ChannelRanker, list_loss


class ChannelRankingTests(unittest.TestCase):
    def test_identity_initialization_and_listwise_gradient(self):
        x = torch.rand(2, 3, 5)
        prior = torch.rand(2, 3)
        selected = torch.tensor([0, 2])
        for rank in (None, 3):
            with self.subTest(rank=rank):
                model = ChannelRanker(torch.zeros(5), torch.ones(5), width=8, cross_rank=rank)
                residual = model(x)
                torch.testing.assert_close(prior + residual, prior)
                gradient, = torch.autograd.grad(
                    list_loss(prior, residual, selected, torch.ones(2), penalty=0.01), residual)
                self.assertTrue((gradient[torch.arange(2), selected] < 0).all())
                self.assertTrue((gradient[~torch.nn.functional.one_hot(selected, 3).bool()] > 0).all())

    def test_calibration_keeps_scores_for_surviving_candidates(self):
        scores = np.array([[1.0, 2, 3], [10, 10, 10]])
        replaced = np.array([[False, False, True], [True, False, False]])
        original, merged = normalize_to_original(scores, replaced, np.array([4.0, 8]))
        np.testing.assert_array_equal(merged[~replaced], original[~replaced])
        np.testing.assert_allclose(merged[replaced], [2 / scores[0].std(), -2])

    def test_click_exposure_and_missing_history_have_separate_signals(self):
        vector = torch.tensor([[0., 0.], [1., 0.], [0., 1.]])
        attr = torch.tensor([[0, 0], [1, 0], [1, 2]])
        size = torch.tensor([[0, 0], [10, 2], [10, 1]])
        value = channel_features(
            torch.tensor([[1, 2]]), torch.tensor([[0, 1, 2]]),
            torch.tensor([[0, 2, 1]]), torch.tensor([[0., 3., 0.]]),
            vector, attr, size, torch.tensor([[True, False]]), 0.04
        )
        torch.testing.assert_close(value[0, 0, :8], torch.tensor([1., 1., 1., 0., 1., 1., 3., 1.]))
        torch.testing.assert_close(value[0, 1, :8], torch.tensor([0., 0., 1., 1., 1., 0., 0., 2.]))
        absent = channel_features(
            torch.tensor([[1]]), torch.tensor([[0]]), torch.tensor([[0]]), torch.tensor([[0.]]),
            vector, attr, size, torch.tensor([[False]]), 0.04
        )
        torch.testing.assert_close(absent, torch.zeros_like(absent))
