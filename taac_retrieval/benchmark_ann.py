"""Measure FlatIP vs HNSW on identical vectors, filters, CPU threads and queries."""

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import time

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from .ann import search_filtered, ann_recall, select_configuration, lift_database, InnerProductFromL2
from .metrics import recommendation_metrics


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def build_index(vectors, kind, output, threads, m=32, ef_construction=100, geometry='raw_ip'):
    faiss.omp_set_num_threads(threads)
    started = time.perf_counter()
    dimension = vectors.shape[1]
    lifted = kind != 'flat' and geometry == 'mips_l2'
    radius_squared = max(float(np.square(vectors[start:start+100000].astype(np.float64)).sum(1).max())
                         for start in range(1, len(vectors), 100000)) if lifted else None
    index = (faiss.IndexFlatIP(dimension) if kind == 'flat' else
             faiss.IndexHNSWFlat(dimension + int(lifted), m,
                                 faiss.METRIC_L2 if lifted else faiss.METRIC_INNER_PRODUCT))
    if kind != 'flat':
        index.hnsw.efConstruction = ef_construction
    # Padding is not indexed. Returned Faiss IDs are mapped back with +1.
    with threadpool_limits(limits=threads):
        for start in range(1, len(vectors), 100000):
            block = np.ascontiguousarray(vectors[start:start + 100000], dtype=np.float32)
            index.add(lift_database(block, radius_squared) if lifted else block)
            if kind != 'flat':
                log(f'HNSW build: {index.ntotal:,}/{len(vectors)-1:,} items')
    build_seconds = time.perf_counter() - started
    faiss.write_index(index, str(output))
    return index, dict(build_seconds=build_seconds, serialized_index_bytes=Path(output).stat().st_size,
                       items=index.ntotal, dimension=dimension, index_dimension=index.d,
                       geometry='mips_l2' if lifted else 'raw_ip', radius_squared=radius_squared)


def latency(index, queries, seen, config):
    threads = config['latency_threads']
    faiss.omp_set_num_threads(threads)
    times = []
    rng = np.random.default_rng(719)
    with threadpool_limits(limits=threads):
        for row in range(min(config['warmup_queries'], len(queries))):
            search_filtered(index, queries[row:row + 1], seen[row:row + 1])
        for _ in range(config['latency_repeats']):
            for row in rng.permutation(len(queries)):
                start = time.perf_counter_ns()
                search_filtered(index, queries[row:row + 1], seen[row:row + 1])
                times.append((time.perf_counter_ns() - start) / 1e6)
    # Batch throughput is a separate 8-thread workload, not the reciprocal of p95.
    batch_times = []
    faiss.omp_set_num_threads(8)
    with threadpool_limits(limits=8):
        search_filtered(index, queries[:32], seen[:32])
        for _ in range(3):
            start = time.perf_counter()
            search_filtered(index, queries, seen)
            batch_times.append(time.perf_counter() - start)
    return dict(single_query_p50_ms=float(np.quantile(times, .5)),
                single_query_p95_ms=float(np.quantile(times, .95)),
                single_query_mean_ms=float(np.mean(times)), measured_single_queries=len(times),
                single_query_threads=threads,
                batch_qps_8_threads=float(len(queries) / np.median(batch_times)),
                batch_queries=len(queries), latency_scope='CPU index search plus Python history filtering; warm cache; excludes encoder/network')


def ranking_metrics(predictions, targets):
    hits, ndcg = recommendation_metrics(predictions, targets)
    return {'HR@10': float(hits.mean()), 'NDCG@10': float(ndcg.mean())}


def benchmark(args):
    export_dir, output = Path(args.export_dir), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'benchmark.json').exists():
        raise FileExistsError('Benchmark results exist. Choose a new output directory.')
    study = json.loads(Path(args.config).read_text())
    config = study['faiss']
    metadata = json.loads((export_dir / 'metadata.json').read_text())
    identity = {key: metadata[key] for key in ['checkpoint_sha256', 'items', 'dimension']}
    identity['hnsw_geometry'] = config.get('hnsw_geometry', 'raw_ip')
    identity_path = output / 'export_identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError('Existing index files belong to another model/export.')
    elif list(output.glob('*.faiss')):
        raise ValueError('Existing indexes have no model identity; choose a new output directory.')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    vectors = np.load(export_dir / 'item_vectors.npy', mmap_mode='r')
    if len(vectors) - 1 != metadata['items'] or not np.isfinite(vectors).all():
        raise ValueError('Invalid item-vector export.')
    val = dict(np.load(export_dir / 'val_queries.npz'))
    take = min(config['latency_queries'], len(val['queries']))
    val_indices = np.sort(np.random.default_rng(719).choice(len(val['queries']), take, replace=False))
    val_queries, val_seen = val['queries'][val_indices], val['seen'][val_indices]
    indexes = {}
    for name in ['flat', 'hnsw']:
        index_path = output / f'{name}.faiss'
        build_path = output / f'{name}_build.json'
        if index_path.exists() and build_path.exists():
            indexes[name] = faiss.read_index(str(index_path))
        else:
            log(f'Building {name} over {metadata["items"]:,} vectors')
            index, description = build_index(vectors, name, index_path, config['build_threads'],
                                              config['M'], config['efConstruction'], config.get('hnsw_geometry', 'raw_ip'))
            build_path.write_text(json.dumps(description, indent=2) + '\n')
            indexes[name] = index
    hnsw_description = json.loads((output / 'hnsw_build.json').read_text())
    searched_hnsw = (InnerProductFromL2(indexes['hnsw'], hnsw_description['radius_squared'])
                     if hnsw_description.get('geometry') == 'mips_l2' else indexes['hnsw'])
    reference, reference_scores = search_filtered(indexes['flat'], val_queries, val_seen)
    log('Measuring warm FlatIP single-query latency')
    flat_latency = latency(indexes['flat'], val_queries, val_seen, config)
    rows = []
    for ef in config['efSearch']:
        indexes['hnsw'].hnsw.efSearch = ef
        faiss.omp_set_num_threads(8)
        with threadpool_limits(limits=8):
            found, _ = search_filtered(searched_hnsw, val_queries, val_seen)
        row = dict(efSearch=ef, **latency(searched_hnsw, val_queries, val_seen, config),
                   **{'ANNRecall@10': ann_recall(found, reference)},
                   validation_recommendation=ranking_metrics(found, val['targets'][val_indices]))
        rows.append(row)
        log(json.dumps(row))
    (output / 'validation_sweep.json').write_text(json.dumps(rows, indent=2) + '\n')
    chosen = select_configuration(rows, config['minimum_validation_ann_recall'])
    # Lock index parameters using validation queries before opening test comparisons.
    lock = dict(selected=chosen, criterion='smallest validation p95 among HNSW settings meeting ANNRecall threshold',
                minimum_validation_ann_recall=config['minimum_validation_ann_recall'],
                recommendation_targets_used_for_selection=False)
    (output / 'index_selection.json').write_text(json.dumps(lock, indent=2) + '\n')
    indexes['hnsw'].hnsw.efSearch = chosen['efSearch']
    faiss.write_index(indexes['hnsw'], str(output / 'hnsw.faiss'))
    test = dict(np.load(export_dir / 'test_queries.npz'))
    test_count = min(config['latency_queries'], len(test['queries']))
    test_indices = np.sort(np.random.default_rng(720).choice(len(test['queries']), test_count, replace=False))
    tq, ts = test['queries'][test_indices], test['seen'][test_indices]
    faiss.omp_set_num_threads(8)
    with threadpool_limits(limits=8):
        exact_test, exact_scores = search_filtered(indexes['flat'], tq, ts)
        approximate, approximate_scores = search_filtered(searched_hnsw, tq, ts)
    # Independent scalar NumPy scores validate raw-IP semantics and row IDs.
    independently_scored = np.einsum('qd,qkd->qk', tq.astype(np.float64), vectors[exact_test].astype(np.float64))
    if not np.allclose(independently_scored, exact_scores, rtol=2e-5, atol=2e-5):
        raise AssertionError('Faiss raw inner-product scores/IDs do not match the exported vectors.')
    result = dict(export=metadata, configuration=config, validation_configurations=rows,
                  flat_validation_latency=flat_latency, selected=chosen,
                  test=dict(queries=test_count, **{'ANNRecall@10': ann_recall(approximate, exact_test)},
                            flat_recommendation=ranking_metrics(exact_test, test['targets'][test_indices]),
                            hnsw_recommendation=ranking_metrics(approximate, test['targets'][test_indices])),
                  test_flat_latency=latency(indexes['flat'], tq, ts, config),
                  test_hnsw_latency=latency(searched_hnsw, tq, ts, config),
                  build={name: json.loads((output / f'{name}_build.json').read_text()) for name in indexes},
                  faiss_version=faiss.__version__, faiss_compile_options=faiss.get_compile_options(),
                  platform=platform.platform(), cpu_count=os.cpu_count(),
                  cpu_model=next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                  if line.startswith('model name')), 'unknown'),
                  ids_and_inner_product_independently_verified=True,
                  caveats=['Latency is warmed index-only CPU timing, not end-to-end production latency.',
                           'Serialized bytes describe persisted index size, not peak resident memory.',
                           'Multithreaded HNSW construction is not guaranteed bitwise reproducible.',
                           'Strict ID-overlap recall can penalize alternative items tied at the top-k boundary.'])
    np.savez_compressed(output / 'test_search.npz', rows=test['rows'][test_indices],
                        exact=exact_test, approximate=approximate, targets=test['targets'][test_indices],
                        validation_query_indices=val_indices, test_query_indices=test_indices)
    (output / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
    log(f'Benchmark complete: {output / "benchmark.json"}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/objective_study.json')
    benchmark(parser.parse_args())
