"""Nested depth initialization preserves historical weights and RNG controls."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from taac_retrieval.data import PreparedData
from taac_retrieval.initialization import prepare_depth_initialization, apply_depth_initialization, ordered_digest
from taac_retrieval.model import SequenceRetriever
from test_retrieval import write_fixture


class DepthInitializationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        write_fixture(self.cache)
        meta = json.loads((self.cache / 'metadata.json').read_text())
        meta['preparation_config'] = {'fixture': True}
        (self.cache / 'metadata.json').write_text(json.dumps(meta))
        self.data = PreparedData(self.cache)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(7).get_state())
            model = SequenceRetriever(self.data, hidden=8, blocks=1, heads=1, dropout=.2)
            self.baseline_rng = torch.get_rng_state().clone()
        self.config = dict(seed=7, hidden=8, blocks=1, heads=1, dropout=.2, variant='baseline',
                           protocol=meta, shared_initialization_sha256=ordered_digest(
                               model.state_dict(), [name for name, _ in model.named_parameters()]))
        self.reference = self.root / 'reference.json'
        self.reference.write_text(json.dumps(self.config))
        self.output = self.root / 'initial.pt'

    def tearDown(self):
        self.directory.cleanup()

    def prepare(self):
        return prepare_depth_initialization(self.cache, self.reference, self.output)

    def test_nested_weights_and_original_hash_are_preserved(self):
        manifest = self.prepare()
        models = [SequenceRetriever(self.data, hidden=8, blocks=depth, heads=1, dropout=.2) for depth in [1, 2, 4]]
        metadata = [apply_depth_initialization(model, self.output, self.data) for model in models]
        for record in metadata:
            self.assertEqual(record['common_initialization_sha256'], self.config['shared_initialization_sha256'])
        for left, right in zip(models, models[1:]):
            a, b = dict(left.named_parameters()), dict(right.named_parameters())
            for name in a:
                torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        self.assertEqual(set(manifest['parameters_by_depth']), {'1', '2', '4'})
        self.assertEqual(manifest, self.prepare())

    def test_generation_preserves_external_rng_and_never_touches_cuda(self):
        before = torch.get_rng_state().clone()
        with patch('torch.cuda.manual_seed_all', side_effect=AssertionError('CUDA seeded')), \
                patch('torch.cuda.get_rng_state', side_effect=AssertionError('CUDA read')):
            self.prepare()
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        model = SequenceRetriever(self.data, hidden=8, blocks=2, heads=1, dropout=.2)
        apply_depth_initialization(model, self.output, self.data)
        torch.testing.assert_close(self.baseline_rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_cache_shape_and_existing_identity_are_checked(self):
        self.prepare()
        with self.assertRaises(ValueError):
            apply_depth_initialization(SequenceRetriever(self.data, hidden=16, blocks=2), self.output, self.data)
        model = SequenceRetriever(self.data, hidden=8, blocks=2, heads=1, dropout=.2)
        self.data.metadata['preparation_config'] = {'different': True}
        with self.assertRaises(ValueError):
            apply_depth_initialization(model, self.output, self.data)
        self.config['seed'] = 8
        self.reference.write_text(json.dumps(self.config))
        with self.assertRaises(ValueError):
            self.prepare()


if __name__ == '__main__':
    unittest.main()
