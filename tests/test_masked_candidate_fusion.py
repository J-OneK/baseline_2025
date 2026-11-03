"""Verify learned-channel seed/policy identity before protected fusion."""

from pathlib import Path
import tempfile
import unittest

from taac_retrieval.coverage import file_digest
from taac_retrieval.masked_candidate_fusion import validate_checkpoint_pair


class MaskedCandidateFusionTests(unittest.TestCase):
    def test_matching_seed_and_checkpoint_are_required(self):
        cfg = dict(seed=2026, variant='baseline', objective='click_infonce', num_negatives=64, temperature=1.,
                   epochs=5, batch_size=128, hidden=32, blocks=1, heads=1, dropout=.2, lr=.001,
                   max_steps=0, skip_test=True)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'weights.pt'
            checkpoint.write_bytes(b'fixed checkpoint fixture')
            a = dict(training=cfg, checkpoint=str(checkpoint), checkpoint_sha256=file_digest(checkpoint))
            b = dict(a, training=dict(cfg, item_id_dropout=.7))
            validate_checkpoint_pair(2026, a, b)
            with self.assertRaisesRegex(ValueError, 'matching training seed'):
                validate_checkpoint_pair(2027, a, b)
            with self.assertRaisesRegex(ValueError, 'mask70'):
                validate_checkpoint_pair(2026, a, dict(b, training=dict(cfg, item_id_dropout=.3)))
            with self.assertRaisesRegex(ValueError, 'recorded checkpoint'):
                validate_checkpoint_pair(2026, a, dict(b, checkpoint_sha256='wrong'))
            with self.assertRaisesRegex(ValueError, 'num_negatives'):
                validate_checkpoint_pair(2026, a, dict(b, training=dict(b['training'], num_negatives=1)))


if __name__ == '__main__':
    unittest.main()
