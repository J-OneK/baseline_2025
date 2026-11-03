"""Regression checks for inner-product semantics, ID offsets and ANN selection."""

from pathlib import Path
from argparse import Namespace
import json
import tempfile
import unittest

import numpy as np

from taac_retrieval.ann import (filter_results, search_filtered, ann_recall, select_configuration,
                                lift_database, InnerProductFromL2)

try:
    import faiss
except ImportError:
    faiss = None


class ANNLogicTests(unittest.TestCase):
    def test_zero_based_index_ids_are_not_padding_or_double_mapped(self):
        found, scores = filter_results(np.array([[0, 1, 2, 3, -1]]),
                                       np.array([[4., 3., 2., 1., -1.]]), np.array([[1, 0]]), k=2)
        np.testing.assert_array_equal(found, [[2, 3]])
        np.testing.assert_array_equal(scores, [[3., 2.]])

    def test_filter_reports_too_few_candidates(self):
        with self.assertRaises(ValueError):
            filter_results(np.array([[0, -1]]), np.zeros((1, 2)), np.array([[1]]), k=1)

    def test_recall_and_selection_do_not_use_recommendation_labels(self):
        self.assertEqual(ann_recall(np.array([[1, 2], [3, 9]]), np.array([[2, 1], [3, 4]])), .75)
        rows = [{'efSearch': 128, 'ANNRecall@10': .9, 'single_query_p95_ms': .1},
                {'efSearch': 256, 'ANNRecall@10': .99, 'single_query_p95_ms': .2},
                {'efSearch': 512, 'ANNRecall@10': 1., 'single_query_p95_ms': .4}]
        self.assertEqual(select_configuration(rows, .98)['efSearch'], 256)
        with self.assertRaises(ValueError):
            select_configuration(rows[:1], .98)


@unittest.skipIf(faiss is None, 'Faiss tests require the project environment')
class FaissIntegrationTests(unittest.TestCase):
    def test_inner_product_is_not_silently_normalized(self):
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[1., 0.], [2., 0.], [.9, .9]], dtype=np.float32))
        found, score = search_filtered(index, np.array([[1., 0.]], dtype=np.float32), np.array([[0]]), k=1)
        self.assertEqual(found[0, 0], 2)
        self.assertEqual(score[0, 0], 2.)

    def test_asymmetric_lift_preserves_ip_scores_and_order(self):
        vectors = np.array([[1., 0.], [2., 0.], [.9, .9]], dtype=np.float32)
        queries = np.array([[1., 0.], [-1., .3]], dtype=np.float32)
        raw, lifted = faiss.IndexFlatIP(2), faiss.IndexFlatL2(3)
        raw.add(vectors)
        lifted.add(lift_database(vectors, 4.))
        expected_scores, expected_ids = raw.search(queries, 3)
        actual_scores, actual_ids = InnerProductFromL2(lifted, 4.).search(queries, 3)
        np.testing.assert_array_equal(actual_ids, expected_ids)
        np.testing.assert_allclose(actual_scores, expected_scores, atol=1e-6)
        with self.assertRaises(ValueError):
            lift_database(vectors, 1.)

    def test_hnsw_persistence_filtering_and_exact_reference(self):
        faiss.omp_set_num_threads(1)
        rng = np.random.default_rng(413)
        vectors, queries = rng.normal(size=(500, 16)).astype(np.float32), rng.normal(size=(20, 16)).astype(np.float32)
        flat = faiss.IndexFlatIP(16)
        hnsw = faiss.IndexHNSWFlat(16, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction, hnsw.hnsw.efSearch = 100, 256
        flat.add(vectors)
        hnsw.add(vectors)
        seen = np.tile(np.arange(1, 11), (20, 1))
        exact, _ = search_filtered(flat, queries, seen)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'index.faiss'
            faiss.write_index(hnsw, str(path))
            loaded = faiss.read_index(str(path))
            approximate, _ = search_filtered(loaded, queries, seen)
        self.assertGreaterEqual(ann_recall(approximate, exact), .99)
        self.assertFalse(np.isin(approximate, np.arange(1, 11)).any())

    def test_benchmark_exports_selection_and_verified_test_scores(self):
        from taac_retrieval.benchmark_ann import benchmark
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exported = root / 'export'
            exported.mkdir()
            rng = np.random.default_rng(517)
            vectors = rng.normal(size=(501, 8)).astype(np.float32)
            vectors[0] = 0
            np.save(exported / 'item_vectors.npy', vectors)
            (exported / 'metadata.json').write_text(json.dumps(dict(items=500, dimension=8, checkpoint_sha256='test-only')))
            for split in ['val', 'test']:
                np.savez(exported / f'{split}_queries.npz', queries=rng.normal(size=(8, 8)).astype(np.float32),
                         seen=np.tile(np.array([0, 1, 2]), (8, 1)), targets=np.full(8, 400), rows=np.arange(8))
            config = root / 'config.json'
            config.write_text(json.dumps(dict(faiss=dict(M=16, efConstruction=40, efSearch=[64, 128], hnsw_geometry='mips_l2',
                               minimum_validation_ann_recall=.9, latency_queries=8, warmup_queries=2,
                               latency_repeats=1, latency_threads=1, build_threads=1))))
            result = benchmark(Namespace(export_dir=str(exported), output=str(root / 'index'), config=str(config)))
            self.assertTrue(result['ids_and_inner_product_independently_verified'])
            self.assertTrue((root / 'index/index_selection.json').exists())
            self.assertGreaterEqual(result['test']['ANNRecall@10'], .9)
