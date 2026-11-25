"""Regression coverage for rare candidate qualification and budget."""
import unittest

import numpy as np

from taac_retrieval.group_recall import RareAttributeIndex, merge_candidates


class GroupRecallTests(unittest.TestCase):
    def test_proposals_require_history_match_and_available_content(self):
        features = np.array([[0, 0], [7, 9], [7, 2], [7, 2], [8, 2],
                             [7, 9], [7, 9], [7, 9]], np.int32)
        content = np.array([[0, 0], [1, 0], [1, 0], [0, 1], [1, 0],
                            [1, 0], [0, 0], [1, 0]], np.float32)
        counts = np.array([0, 4, 0, 1, 1, 6, 0, 0])
        index = RareAttributeIndex(features, content, counts, [0, 1], 20)
        history = np.array([0, 1, 7])
        actions = np.array([0, 2, 3])
        proposals = index.propose(history, actions, 3, 3, .04)
        for mode in proposals:
            np.testing.assert_array_equal(mode[0], [2, 0, 0])
            np.testing.assert_array_equal(mode[1], [3, 0, 0])
        self.assertFalse(index.propose(history, np.array([0, 3, 3]), 3, 3, .04).any())

    def test_budget_protects_prefix_and_existing_tail(self):
        original = np.array([1, 2, 3, 4, 5, 6])
        protected = np.array([True, False, False, False, False, True])
        counts = np.array([0, 10, 20, 4, 7, 12, 30, 0, 2, 0])
        proposals = np.array([[7, 7, 9], [3, 8, 0]])
        merged = merge_candidates(original, protected, counts, proposals, [1, 1])
        np.testing.assert_array_equal(merged, [1, 2, 3, 8, 7, 6])
        self.assertEqual(len(np.unique(merged)), len(original))

    def test_zero_feature_value_and_oversized_postings_do_not_route(self):
        attrs = np.array([[0], [0], [0], [5], [5], [5]], np.int32)
        content = np.array([[0, 0]] + [[1, 0]] * 5, np.float32)
        counts = np.zeros(6, np.int32)
        index = RareAttributeIndex(attrs, content, counts, [0], 2)
        result = index.propose(np.array([1, 3]), np.array([2, 1]), 3, 2, .04)
        self.assertFalse(result.any())


if __name__ == "__main__":
    unittest.main()
