import unittest

import torch

from taac_retrieval.sequence_ranker import HistoryBlock, HistoryRanker


class HistoryRankerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.features = torch.randn(2, 5, 15)
        self.items = torch.randn(2, 5, 7)
        self.history = torch.randn(2, 4, 7)
        self.actions = torch.tensor([[0, 1, 1, 2], [0, 0, 0, 0]])
        self.ages = torch.rand(2, 4)

    def test_zero_initialization_and_masked_history(self):
        for name in ("history_attention", "joint_transformer"):
            model = HistoryRanker(name, 12, 3, 4, torch.zeros(15), torch.ones(15), 7)
            inputs = (self.features, self.items, self.history, self.actions, self.ages)
            torch.testing.assert_close(model(*inputs), torch.zeros(2, 5))
            with torch.no_grad():
                model.score.weight.normal_()
                modified = self.history.clone()
                modified[self.actions == 0] = 99
                expected = model(*inputs)
                self.assertTrue(torch.isfinite(expected).all())
                actual = model(self.features, self.items, modified, self.actions, self.ages)
                torch.testing.assert_close(expected, actual, atol=2e-5, rtol=1e-5)
                order = torch.tensor([4, 0, 2, 1, 3])
                reordered = model(self.features[:, order], self.items[:, order], *inputs[2:])
                torch.testing.assert_close(expected[:, order], reordered, atol=2e-5, rtol=1e-5)

    def test_prefix_encoding_is_causal(self):
        block = HistoryBlock(12, 3)
        sequence = torch.randn(1, 4, 12)
        valid = torch.tensor([[False, True, True, True]])
        modified = sequence.clone()
        modified[:, -1] += torch.arange(12)
        original = block(sequence, valid)[0]
        after = block(modified, valid)[0]
        torch.testing.assert_close(original[:, :-1], after[:, :-1])
        self.assertFalse(torch.equal(original[:, -1], after[:, -1]))
