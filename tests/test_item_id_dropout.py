"""Item masking must simulate OOV without changing content, IDs, or RNG controls."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from taac_retrieval.data import PreparedData
from taac_retrieval.model import SequenceRetriever
from taac_retrieval.train import initialization_hash
from test_retrieval import write_fixture


class ItemIDDropoutTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        write_fixture(self.root)
        self.data = PreparedData(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def model(self, probability):
        torch.manual_seed(41)
        return SequenceRetriever(self.data, dropout=0, item_id_dropout=probability)

    def test_disabled_mask_preserves_initialization_and_rng(self):
        plain, masked = self.model(0), self.model(.7)
        self.assertEqual(initialization_hash(plain), initialization_hash(masked))
        generator = torch.Generator().manual_seed(9)
        before = generator.get_state().clone()
        plain.refresh_item_id_mask(generator)
        torch.testing.assert_close(before, generator.get_state())
        with self.assertRaises(ValueError):
            self.model(1.1)

    def test_one_item_decision_is_shared_and_dedicated_rng_is_used(self):
        model = self.model(.7)
        generator = torch.Generator().manual_seed(9)
        global_before = torch.random.get_rng_state().clone()
        model.refresh_item_id_mask(generator)
        torch.testing.assert_close(global_before, torch.random.get_rng_state())
        ids = torch.tensor([[1, 2, 1], [2, 1, 2]])
        first, second = model.encode_items(ids), model.encode_items(ids)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first[0, 0], first[1, 1])
        np.testing.assert_array_equal(model.item_id_lookup.numpy(), np.load(self.root / 'item_id_lookup.npy'))

    def test_full_mask_retains_content_and_eval_restores_full_ids(self):
        plain, masked = self.model(0), self.model(1)
        ids = torch.tensor([0, 1, 2])
        with self.assertRaises(RuntimeError):
            masked.encode_items(ids)
        masked.refresh_item_id_mask(torch.Generator().manual_seed(2))
        with torch.no_grad():
            original = masked.encode_items(ids)
            masked.item_emb.weight[1:].add_(50)
            torch.testing.assert_close(masked.encode_items(ids), original)
            masked.item_mm81[1].add_(100)
            self.assertFalse(torch.allclose(masked.encode_items(ids)[1], original[1]))
            torch.testing.assert_close(masked.encode_items(ids)[0], torch.zeros(32))
        masked.load_state_dict(plain.state_dict())
        masked.item_mm81.copy_(plain.item_mm81)
        masked.eval()
        torch.testing.assert_close(masked.encode_items(ids), plain.encode_items(ids))
        self.assertEqual(set(plain.state_dict()), set(masked.state_dict()))


if __name__ == '__main__':
    unittest.main()
