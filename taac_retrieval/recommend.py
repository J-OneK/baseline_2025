"""One user's causal history -> model vector -> persisted Faiss HNSW -> ad OIDs."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import faiss
import numpy as np
import torch

from .ann import search_filtered, InnerProductFromL2
from .data import SequenceDataset
from .evaluate_checkpoint import load_model
from .metrics import to_device


@torch.inference_mode()
def recommend(args):
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(1)
    model, data, checkpoint = load_model(args.checkpoint, args.cache, args.device)
    exported = json.loads((Path(args.export_dir) / 'metadata.json').read_text())
    digest = hashlib.sha256()
    with open(args.checkpoint, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != exported['checkpoint_sha256']:
        raise ValueError('The query model differs from the indexed item model.')
    indexed_identity = json.loads((Path(args.index_dir) / 'export_identity.json').read_text())
    if indexed_identity['checkpoint_sha256'] != exported['checkpoint_sha256']:
        raise ValueError('This index directory belongs to a different model.')
    index = faiss.read_index(str(Path(args.index_dir) / 'hnsw.faiss'))
    description = json.loads((Path(args.index_dir) / 'hnsw_build.json').read_text())
    if description.get('geometry') == 'mips_l2':
        index = InnerProductFromL2(index, description['radius_squared'])
    if index.ntotal != data.num_items or index.d != model.hidden:
        raise ValueError('Index dimension/catalog do not match the checkpoint.')
    sample = SequenceDataset(data, args.split)[args.index]
    batch = to_device({key: torch.as_tensor(value).unsqueeze(0) for key, value in sample.items()}, args.device)
    # Warm up once; report request components separately from process startup.
    model.predict(batch)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize(args.device)
    started = time.perf_counter()
    query = model.predict(batch).cpu().numpy().astype(np.float32, copy=False)
    encoded = time.perf_counter()
    found, scores = search_filtered(index, query, sample['items'][None])
    finished = time.perf_counter()
    result = dict(user_rid=int(data.user_rids[sample['row']]),
                  recommendations=[dict(item_oid=int(data.catalog_oids[item]), score=float(score))
                                   for item, score in zip(found[0], scores[0])],
                  timings_ms=dict(encoder_and_transfer=(encoded - started) * 1000,
                                  index_and_history_filter=(finished - encoded) * 1000,
                                  warm_request_total=(finished - started) * 1000),
                  timing_note='One warmed local request; startup/model/index loading excluded; not a p95 measurement.',
                  offline_target_oid=int(data.catalog_oids[sample['target']]))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--export-dir', required=True)
    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--index', type=int, default=0)
    recommend(parser.parse_args())
