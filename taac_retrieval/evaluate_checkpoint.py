"""Evaluate a validation-selected checkpoint once on its frozen test cohort."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import PreparedData
from .metrics import evaluate, evaluate_popularity
from .model import SequenceRetriever


def load_model(checkpoint_path, cache, device):
    data = PreparedData(cache)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['cache_signature'] != data.metadata['preparation_config']:
        raise ValueError('Checkpoint cache signature does not match.')
    if 'evaluation_indices' in checkpoint:
        data.set_eval_indices(checkpoint['evaluation_indices'].numpy())
    config = checkpoint['config']
    model = SequenceRetriever(data, config['hidden'], config['blocks'], config['heads'], config['dropout'],
                              use_actions=config['variant'] == 'actions')
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), data, checkpoint


def evaluate_checkpoint(args):
    output = Path(args.output)
    if output.with_suffix('.json').exists():
        raise FileExistsError('This evaluation already exists; do not silently overwrite a test result.')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model, data, checkpoint = load_model(args.checkpoint, args.cache, args.device)
    metrics, per_user = evaluate(model, data, args.split, args.device)
    popularity, popularity_per_user = evaluate_popularity(data, args.split)
    np.savez_compressed(output.with_suffix('.npz'), **per_user)
    np.savez_compressed(output.parent / f'{output.name}_popularity.npz', **popularity_per_user)
    digest = hashlib.sha256()
    with open(args.checkpoint, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    result = dict(metrics=metrics, popularity=popularity, split=args.split,
                  selected_epoch=checkpoint['epoch'], validation=checkpoint['validation'],
                  checkpoint_sha256=digest.hexdigest(), checkpoint=str(Path(args.checkpoint).resolve()),
                  training=checkpoint['config'],
                  evaluation_cohort_sha256=hashlib.sha256(np.asarray(data.eval_indices, dtype=np.int64).tobytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True, help='Output prefix without extension')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', default='cuda:0')
    evaluate_checkpoint(parser.parse_args())
