"""Export a frozen checkpoint into item vectors and real validation/test queries."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SequenceDataset
from .evaluate_checkpoint import load_model
from .metrics import to_device


@torch.inference_mode()
def export(args):
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with open(args.checkpoint, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    checkpoint_sha = digest.hexdigest()
    if (output / 'metadata.json').exists():
        existing = json.loads((output / 'metadata.json').read_text())
        if existing['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError('The export already belongs to another checkpoint.')
        return existing
    model, data, checkpoint = load_model(args.checkpoint, args.cache, args.device)
    vectors = np.lib.format.open_memmap(output / 'item_vectors.npy', mode='w+', dtype=np.float32,
                                        shape=(data.num_items + 1, model.hidden))
    for start in range(0, len(vectors), 8192):
        end = min(start + 8192, len(vectors))
        vectors[start:end] = model.encode_items(torch.arange(start, end, device=args.device)).cpu().numpy()
    vectors.flush()
    for split in ['val', 'test']:
        queries, seen, targets, rows = [], [], [], []
        loader = DataLoader(SequenceDataset(data, split), batch_size=256, shuffle=False)
        for batch in loader:
            queries.append(model.predict(to_device(batch, args.device)).cpu().numpy())
            seen.append(batch['items'].numpy())
            targets.append(batch['target'].numpy())
            rows.append(batch['row'].numpy())
        np.savez_compressed(output / f'{split}_queries.npz', queries=np.concatenate(queries),
                            seen=np.concatenate(seen), targets=np.concatenate(targets), rows=np.concatenate(rows))
    metadata = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=checkpoint_sha,
                    cache=str(Path(args.cache).resolve()), cache_signature=checkpoint['cache_signature'],
                    cohort_sha256=hashlib.sha256(np.asarray(data.eval_indices, dtype=np.int64).tobytes()).hexdigest(),
                    items=data.num_items, dimension=model.hidden, dtype='float32',
                    metric='raw inner product; no vector normalization', index_id_to_local_id_offset=1,
                    evaluation_users=len(data.eval_indices), seed=checkpoint['config']['seed'])
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    export(parser.parse_args())
