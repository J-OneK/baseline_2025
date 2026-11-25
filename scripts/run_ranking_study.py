"""Register four matched ranking runs on a frozen retrieval candidate pool."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import os,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from taac_retrieval.ranking_data import read,write_new
from taac_retrieval.ranking_model import digest


def freeze(path,value):
    path=Path(path)
    if path.exists():assert read(path)==value,path
    else:write_new(path,value)


def job(config,variant,seed,gpu):
    root=REPO/config['root'];folder=root/'training'/f'{variant}_seed{seed}'
    if (folder/'result.json').exists():return
    command=[sys.executable,'-m','taac_retrieval.ranking_train','--variant',variant,'--seed',str(seed),'--device',f'cuda:{gpu}']
    freeze(root/f'{variant}_seed{seed}.command.json',dict(command=command))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
    print(f'Started {variant} seed{seed} GPU{gpu}',flush=True)
    with (root/f'{variant}_seed{seed}.log').open('x') as log:
        subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result=read(folder/'result.json');r=result['validation']['ranking']
    print(f'Completed {variant} seed{seed}: HR10={r["HR10"]} NDCG10={r["NDCG10"]}',flush=True)


def main():
    config=read(REPO/'configs/ranking_study.json');root=REPO/config['root']
    assert config['model']==dict(hidden=32,mlp=[128,64],attention=[64,32],dropout=.1)
    examples=root/'examples';meta=read(examples/'metadata.json');assert meta['configuration']==config['data']
    identity=dict(examples={p.name:digest(p) for p in examples.glob('*.npy')},features={})
    for seed in config['seeds']:
        folder=root/f'features_seed{seed}';f=read(folder/'metadata.json')
        assert f['cache_metadata_sha256']==digest(Path(config['cache'])/'metadata.json')
        assert f['examples_metadata_sha256']==digest(examples/'metadata.json')
        assert f['checkpoint_sha256']==digest(f['checkpoint'])
        for name,sha in f['files'].items():assert digest(folder/name)==sha
        assert f['candidates_sha256']==digest(folder/'validation_candidates.npz')
        identity['features'][str(seed)]=dict(metadata_sha256=digest(folder/'metadata.json'),checkpoint_sha256=f['checkpoint_sha256'])
    names=['configs/ranking_study.json','scripts/run_ranking_study.py','tests/test_ranking.py','model.py',
           'taac_retrieval/data.py','taac_retrieval/model.py','taac_retrieval/metrics.py','taac_retrieval/train.py',
           'taac_retrieval/coverage.py','taac_retrieval/evaluate_checkpoint.py']
    names+=[str(p.relative_to(REPO)) for p in (REPO/'taac_retrieval').glob('ranking_*.py')]
    freeze(root/'plan.json',config);freeze(root/'identity.json',identity)
    freeze(root/'source_manifest.json',{n:digest(REPO/n) for n in sorted(names)})
    if not (root/'registered.json').exists():
        freeze(root/'registered.json',dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),test_evaluated=False))
    for seed in config['seeds']:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(job,config,variant,seed,gpu) for variant,gpu in [('mean_mlp',5),('din_attention',6)]]
            for f in futures:f.result()
    for name,sha in read(root/'source_manifest.json').items():assert digest(REPO/name)==sha,name
    freeze(root/'training_complete.json',dict(runs=4,test_evaluated=False,source_unchanged=True))


if __name__=='__main__':
    main()
