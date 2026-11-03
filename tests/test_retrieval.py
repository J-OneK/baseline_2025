"""Checks for leakage, actual feature use, ranking, and the action ablation."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from dataset import MyDataset, MyTestDataset, load_seq_as_list
from infer import parse_candidate_feature
from taac_retrieval.data import PreparedData, SequenceDataset, make_input, sample_negatives
from taac_retrieval.metrics import exact_topk, recommendation_metrics
from taac_retrieval.model import SequenceRetriever
from taac_retrieval.prepare import clean_and_split


def write_fixture(root):
    rng = np.random.default_rng(41)
    metadata = dict(item_fields=['100'], user_fields=['103'], user_array_fields=['106'],
                    feature_sizes={'100': 4, '103': 3, '106': 3})
    (root / 'metadata.json').write_text(json.dumps(metadata))
    counts = np.zeros(31, dtype=np.int64)
    counts[[1, 2, 3, 7, 8, 9]] = 1
    lookup = np.zeros(31, dtype=np.int32)
    lookup[counts > 0] = np.arange(1, 7)
    arrays = dict(items=np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=np.int32),
                  actions=np.array([[1, 2, 1, 2, 1, 2]] * 2, dtype=np.uint8),
                  timestamps=np.array([[1, 2, 3, 4, 5, 6]] * 2),
                  user_rids=np.array([51, 99]), lengths=np.array([6, 6]),
                  train_cut=np.array([3, 3]), val_cut=np.array([3, 3]), test_cut=np.array([5, 5]),
                  val_pos=np.array([3, 3]), test_pos=np.array([5, 5]), train_counts=counts,
                  eval_indices=np.array([0, 1]), catalog_rids=np.arange(31), catalog_oids=np.arange(31) + 1000,
                  item_sparse=(np.arange(31) % 4).astype(np.int32)[:, None],
                  item_mm81=rng.normal(size=(31, 32)).astype(np.float32),
                  item_id_lookup=lookup, user_sparse=np.array([[0], [1], [2]], dtype=np.int32),
                  user_array_106=np.array([[0, 0], [1, 2], [2, 0]], dtype=np.int32))
    arrays['item_sparse'][0] = 0
    arrays['item_mm81'][0] = 0
    for name, value in arrays.items():
        np.save(root / f'{name}.npy', value)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        write_fixture(self.root)
        self.data = PreparedData(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_chronological_holdout_has_no_target_or_future_history(self):
        events = [dict(item_id=i + 1, timestamp=i + 10, action_type=1 if i in [2, 5, 8] else 0)
                  for i in range(10)]
        split = clean_and_split(events[::-1], min_history=2)
        self.assertEqual((split['val_pos'], split['test_pos']), (5, 8))
        self.assertLess(split['times'][split['train_cut'] - 1], split['times'][split['val_pos']])
        self.assertNotIn(split['items'][split['val_pos']], split['items'][:split['val_cut']])
        self.assertNotIn(split['items'][split['test_pos']], split['items'][:split['test_cut']])

    def test_previously_exposed_ad_is_not_a_holdout_target(self):
        events = [dict(item_id=i, timestamp=i, action_type=1 if i in [3, 6, 9] else 0)
                  for i in range(1, 11)]
        events.insert(1, dict(item_id=6, timestamp=1, action_type=0))
        split = clean_and_split(events, min_history=1)
        self.assertEqual(split['items'][split['val_pos']], 3)
        self.assertEqual(split['items'][split['test_pos']], 9)

    def test_timestamp_ties_are_excluded_and_unknown_actions_are_preserved(self):
        events = [dict(item_id=i + 1, timestamp=t, action_type=a)
                  for i, (t, a) in enumerate([(1, None), (2, 0), (3, 0), (3, 1), (4, 0), (5, 1)])]
        split = clean_and_split(events, min_history=2)
        self.assertEqual(split['val_cut'], 2)
        self.assertEqual(split['actions'][0], 3)

    def test_training_samples_use_only_prefix(self):
        sample = SequenceDataset(self.data, 'train')[0]
        self.assertEqual(sample['positive'][sample['positive'] > 0].tolist(), [1, 2, 3])
        self.assertNotIn(4, sample['items'])
        self.assertNotIn(6, sample['items'])

    def test_eval_keeps_last_observed_event(self):
        sample = SequenceDataset(self.data, 'test')[0]
        self.assertEqual(sample['items'][-1], 5)
        self.assertEqual(sample['target'], 6)
        seq, kinds, actions = make_input([3, 4], [1, 2], 6)
        self.assertEqual(seq[-2:].tolist(), [3, 4])
        self.assertEqual(kinds[-3:].tolist(), [2, 1, 1])

    def test_negative_sampling_excludes_training_history(self):
        pool = torch.arange(1, 20)
        seen = torch.tensor([[1, 2, 3], [4, 5, 6]])
        negative = sample_negatives(pool, seen, (2, 100), torch.Generator().manual_seed(5))
        self.assertFalse((negative.unsqueeze(-1) == seen.unsqueeze(1)).any())

    def test_negative_pool_exhaustion_is_reported(self):
        with self.assertRaises(ValueError):
            sample_negatives(torch.tensor([1, 2]), torch.tensor([[1, 2]]), (1, 2), torch.Generator())

    def test_metric_known_ranks(self):
        hits, ndcg = recommendation_metrics([[1, 2, 3], [3, 2, 1], [1, 2, 3]], [1, 1, 8])
        np.testing.assert_allclose(hits, [1, 1, 0])
        np.testing.assert_allclose(ndcg, [1, .5, 0])

    def test_chunked_search_matches_exhaustive_inner_product(self):
        torch.manual_seed(9)
        queries, vectors = torch.randn(3, 8), torch.randn(40, 8)
        seen = torch.tensor([[1, 2, 0], [8, 13, 0], [4, 12, 0]])
        scores = queries @ vectors.T
        scores[:, 0] = -torch.inf
        scores.scatter_(1, seen, -torch.inf)
        expected = scores.topk(10).indices
        actual = exact_topk(queries, vectors, seen, chunk_size=7)
        torch.testing.assert_close(actual, expected)

    def test_content_oid_join_and_user_token_separation(self):
        dataset = object.__new__(MyDataset)
        dataset.feature_types = {'item_emb': ['81']}
        dataset.feature_default_value = {'81': np.zeros(32, dtype=np.float32)}
        dataset.indexer_i_rev = {1: 20000000029}
        vector = np.ones(32, dtype=np.float32)
        dataset.mm_emb_dict = {'81': {'20000000029': vector}}
        np.testing.assert_array_equal(dataset.fill_missing_feat({}, 1)['81'], vector)
        np.testing.assert_array_equal(dataset.fill_missing_feat({}, 1, is_item=False)['81'], np.zeros(32))

    def test_downloaded_candidate_schema_and_cold_start(self):
        self.assertEqual(parse_candidate_feature({'cold_start': 0, 'feature_value': '42'}), 42)
        self.assertEqual(parse_candidate_feature({'cold_start': 1, 'feature_value': '123456'}), 0)
        self.assertEqual(parse_candidate_feature({'cold_start': None, 'feature_value': None}), 0)
        self.assertEqual(parse_candidate_feature(None), 0)

    def test_legacy_arrow_batch_offsets_and_null_actions(self):
        folder = self.root / 'parquet'
        folder.mkdir()
        rows = [{'user_id': 42, 'seq': [dict(item_id=10, action_type=None, timestamp=1)]},
                {'user_id': 99, 'seq': [dict(item_id=20, action_type=0, timestamp=2),
                                       dict(item_id=30, action_type=1, timestamp=3)]}]
        pq.write_table(pa.Table.from_pylist(rows), folder / 'part.parquet')
        events, offsets = load_seq_as_list(str(folder), batch_size=1)
        self.assertEqual(offsets[42], (0, 1))
        self.assertEqual(offsets[99], (1, 2))
        self.assertEqual(events['item_id'].tolist(), [10, 20, 30])
        self.assertEqual(events['action_type'].tolist(), [-1, 0, 1])

    def test_legacy_test_input_keeps_final_event_for_noncontiguous_user_id(self):
        dataset = object.__new__(MyTestDataset)
        dataset.user_ids, dataset.maxlen, dataset.itemnum = [42], 5, 100
        dataset.indexer_u_rev = {42: 'user_42'}
        dataset.feature_default_value = {'100': 0}
        dataset.new_load_user_data = lambda uid: [
            (uid, 10, None, {'100': 1}, 0, 1),
            (uid, 20, None, {'100': 2}, 1, 2),
            (uid, None, {'103': 1}, None, None, 2)]
        dataset.fill_missing_feat = lambda feat, item_id, is_item=True: feat
        sequence, types, features, user = dataset[0]
        self.assertEqual(sequence[-1], 20)
        self.assertEqual(types[-1], 1)
        self.assertEqual(user, 'user_42')

    def test_only_action_variant_responds_to_action_changes(self):
        torch.manual_seed(17)
        baseline = SequenceRetriever(self.data, hidden=8, dropout=0).eval()
        torch.manual_seed(17)
        actions = SequenceRetriever(self.data, hidden=8, dropout=0, use_actions=True).eval()
        for key in baseline.state_dict():
            torch.testing.assert_close(baseline.state_dict()[key], actions.state_dict()[key])
        batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in SequenceDataset(self.data, 'test')[0].items()}
        altered = dict(batch, actions=batch['actions'].clone())
        altered['actions'][batch['kinds'] == 1] = 2
        with torch.no_grad():
            torch.testing.assert_close(baseline.predict(batch), baseline.predict(altered))
            self.assertFalse(torch.allclose(actions.predict(batch), actions.predict(altered)))

    def test_causal_mask_and_padding_have_finite_gradients(self):
        torch.manual_seed(19)
        model = SequenceRetriever(self.data, hidden=8, dropout=0, use_actions=True)
        sample = SequenceDataset(self.data, 'test')[0]
        batch = {k: torch.as_tensor(v).unsqueeze(0) for k, v in sample.items()}
        states = model.encode_history(batch['items'], batch['kinds'], batch['actions'], batch['user'])
        altered = batch['items'].clone()
        altered[:, -1] = 15
        other = model.encode_history(altered, batch['kinds'], batch['actions'], batch['user'])
        torch.testing.assert_close(states[:, :-1], other[:, :-1])
        self.assertTrue(torch.isfinite(states).all())
        states.square().sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
