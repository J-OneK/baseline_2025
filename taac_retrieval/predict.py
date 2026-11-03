"""Inspect a held-out user's exact top-10 using a trained local checkpoint."""

import argparse
import json

import numpy as np
import torch

from .data import PreparedData, SequenceDataset
from .metrics import exact_topk, to_device
from .model import SequenceRetriever
from .evaluate_checkpoint import load_model


@torch.inference_mode()
def predict(args):
    torch.set_num_threads(4)
    model, data, checkpoint = load_model(args.checkpoint, args.cache, args.device)
    dataset = SequenceDataset(data, args.split)
    sample = dataset[args.index]
    batch = to_device({key: torch.as_tensor(value).unsqueeze(0) for key, value in sample.items()}, args.device)
    vectors = torch.empty(data.num_items + 1, model.hidden, device=args.device)
    for start in range(0, len(vectors), 8192):
        end = min(start + 8192, len(vectors))
        vectors[start:end] = model.encode_items(torch.arange(start, end, device=args.device))
    query = model.predict(batch)
    selected = exact_topk(query, vectors, batch['items'])[0]
    scores = (vectors[selected] * query).sum(-1).cpu().tolist()
    items = selected.cpu().numpy()
    target = sample['target']
    result = dict(user_rid=int(data.user_rids[sample['row']]), split=args.split,
                  target_oid=int(data.catalog_oids[target]), candidates=data.num_items,
                  history_length=int((sample['kinds'] == 1).sum()),
                  top10=[dict(item_oid=int(data.catalog_oids[item]), score=score, is_target=bool(item == target))
                         for item, score in zip(items, scores)])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache', default='artifacts/day1_3/data')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--index', type=int, default=0, help='Index within the fixed evaluation cohort')
    predict(parser.parse_args())
