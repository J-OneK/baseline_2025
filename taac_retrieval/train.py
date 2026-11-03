"""Train matched baseline/action models; select by validation and test once."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PreparedData, SequenceDataset, sample_negatives
from .metrics import evaluate, evaluate_popularity, to_device
from .model import SequenceRetriever
from .objectives import supervision_weights, weighted_bce, sampled_info_nce


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def initialization_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not name.startswith('action_emb.'):
            digest.update(name.encode())
            digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


def train(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise FileExistsError('This run already has results; choose a new --output directory.')
    seed_everything(args.seed)
    data = PreparedData(args.cache)
    if args.eval_indices:
        data.set_eval_indices(np.load(args.eval_indices, allow_pickle=False))
    if args.num_negatives < 1 or args.temperature <= 0:
        raise ValueError('Negatives and temperature must be positive.')
    config = vars(args).copy()
    config.update(cache=str(data.root.resolve()), protocol=data.metadata,
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  device_name=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'CPU',
                  selection_rule='best validation NDCG@10, then HR@10; never select using test',
                  evaluation_cohort_sha256=hashlib.sha256(np.asarray(data.eval_indices, dtype=np.int64).tobytes()).hexdigest(),
                  loss=dict(objective=args.objective, negatives=args.num_negatives,
                            exposure_weight=args.exposure_weight, temperature=args.temperature,
                            sampling='uniform train-visible items, excluding the user training prefix'))
    model = SequenceRetriever(data, args.hidden, args.blocks, args.heads, args.dropout,
                              use_actions=args.variant == 'actions')
    config['shared_initialization_sha256'] = initialization_hash(model)
    config['parameters'] = sum(p.numel() for p in model.parameters())
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    generator = torch.Generator().manual_seed(args.seed)
    negative_generator = torch.Generator(device=args.device).manual_seed(args.seed + 17)
    loader = DataLoader(SequenceDataset(data), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=args.device.startswith('cuda'), generator=generator)
    negative_pool = torch.tensor(np.flatnonzero(data.train_counts), dtype=torch.long, device=args.device)
    best = (-1.0, -1.0)
    history, global_step = [], 0
    started = time.monotonic()
    with (output / 'events.jsonl').open('a', buffering=1) as event_file:
        def emit(event):
            event.update(elapsed_seconds=round(time.monotonic() - started, 3))
            line = json.dumps(event)
            print(line, flush=True)
            event_file.write(line + '\n')

        emit(dict(event='start', variant=args.variant, users=data.num_users, candidates=data.num_items,
                  shared_initialization_sha256=config['shared_initialization_sha256']))
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_started = time.monotonic()
            loss_sum, positions = 0.0, 0
            for step, cpu_batch in enumerate(loader):
                batch = to_device(cpu_batch, args.device)
                optimizer.zero_grad(set_to_none=True)
                weights = supervision_weights(batch['target_actions'], args.objective, args.exposure_weight)
                mask = (batch['positive'] != 0) & (weights > 0)
                count = int(mask.sum())
                if count == 0:
                    continue
                if args.num_negatives == 1 and args.objective not in ['click_infonce', 'all_infonce']:
                    # Keep the original one-negative BCE computation and sampling intact.
                    negative = sample_negatives(negative_pool, batch['seen'], batch['positive'].shape,
                                                negative_generator)
                    positive_logits, negative_logits = model(batch, negative)
                    if args.objective == 'all_bce':
                        loss = (F.binary_cross_entropy_with_logits(positive_logits[mask],
                                                                   torch.ones_like(positive_logits[mask])) +
                                F.binary_cross_entropy_with_logits(negative_logits[mask],
                                                                   torch.zeros_like(negative_logits[mask])))
                    else:
                        loss = weighted_bce(positive_logits[mask], negative_logits[mask], weights[mask])
                else:
                    states = model.encode_history(batch['items'], batch['kinds'], batch['actions'], batch['user'])[mask]
                    positive = batch['positive'][mask]
                    row_indices = torch.where(mask)[0]
                    negative = sample_negatives(negative_pool, batch['seen'][row_indices],
                                                (count, args.num_negatives), negative_generator)
                    positive_logits = (states * model.encode_items(positive)).sum(-1)
                    negative_logits = torch.einsum('md,mkd->mk', states, model.encode_items(negative))
                    loss = (sampled_info_nce(positive_logits, negative_logits, args.temperature)
                            if args.objective in ['click_infonce', 'all_infonce'] else
                            weighted_bce(positive_logits, negative_logits, weights[mask]))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at epoch={epoch}, step={step}')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach()) * count
                positions += count
                global_step += 1
                if step % 100 == 0:
                    emit(dict(event='train', epoch=epoch, step=step, global_step=global_step,
                              loss=float(loss.detach()), grad_norm=float(grad_norm)))
                if args.max_steps and global_step >= args.max_steps:
                    break
            training_seconds = time.monotonic() - epoch_started
            validation, per_user = evaluate(model, data, 'val', args.device,
                                            args.eval_batch_size, args.candidate_chunk)
            np.savez_compressed(output / f'validation_epoch{epoch}.npz', **per_user)
            row = dict(event='epoch', epoch=epoch, train_loss=loss_sum / max(positions, 1),
                       supervised_positions=positions, training_seconds=training_seconds, validation=validation)
            history.append(row)
            emit(row)
            score = validation['NDCG@10'], validation['HR@10']
            if score > best:
                best = score
                torch.save(dict(model=model.state_dict(), epoch=epoch, config=vars(args),
                                cache_signature=data.metadata['preparation_config'], validation=validation,
                                evaluation_indices=torch.as_tensor(np.asarray(data.eval_indices).copy())),
                           output / 'best.pt')
            (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
            if args.max_steps and global_step >= args.max_steps:
                break
        checkpoint = torch.load(output / 'best.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        test, popularity = None, None
        if not args.skip_test:
            test, per_user = evaluate(model, data, 'test', args.device, args.eval_batch_size, args.candidate_chunk)
            np.savez_compressed(output / 'test_predictions.npz', **per_user)
            popularity, popularity_per_user = evaluate_popularity(data, 'test')
            np.savez_compressed(output / 'popularity_test_predictions.npz', **popularity_per_user)
        result = dict(variant=args.variant, seed=args.seed, objective=args.objective,
                      num_negatives=args.num_negatives, best_epoch=checkpoint['epoch'],
                      validation=checkpoint['validation'], test=test, popularity_test=popularity,
                      global_steps=global_step, training_users=data.num_users, candidates=data.num_items,
                      shared_initialization_sha256=config['shared_initialization_sha256'],
                      elapsed_seconds=time.monotonic() - started,
                      peak_gpu_memory_mb=torch.cuda.max_memory_allocated(args.device) / 1024**2
                      if args.device.startswith('cuda') else None)
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        emit(dict(event='complete', **result))
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--cache', default='artifacts/baseline/data')
    result.add_argument('--output', required=True)
    result.add_argument('--variant', choices=['baseline', 'actions'], default='baseline')
    result.add_argument('--seed', type=int, default=2026)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--epochs', type=int, default=5)
    result.add_argument('--batch-size', type=int, default=128)
    result.add_argument('--eval-batch-size', type=int, default=256)
    result.add_argument('--candidate-chunk', type=int, default=32768)
    result.add_argument('--workers', type=int, default=2)
    result.add_argument('--hidden', type=int, default=32)
    result.add_argument('--blocks', type=int, default=1)
    result.add_argument('--heads', type=int, default=1)
    result.add_argument('--dropout', type=float, default=0.2)
    result.add_argument('--lr', type=float, default=0.001)
    result.add_argument('--max-steps', type=int, default=0, help='Smoke runs only; 0 trains complete epochs.')
    result.add_argument('--objective', choices=['all_bce', 'weighted_bce', 'click_bce', 'click_infonce', 'all_infonce'], default='click_infonce')
    result.add_argument('--num-negatives', type=int, default=64)
    result.add_argument('--exposure-weight', type=float, default=0.1)
    result.add_argument('--temperature', type=float, default=1.0)
    result.add_argument('--skip-test', action='store_true', help='Development runs must only read validation targets.')
    result.add_argument('--eval-indices', default=None, help='Frozen cohort .npy file; does not change training data.')
    return result


if __name__ == '__main__':
    train(parser().parse_args())
