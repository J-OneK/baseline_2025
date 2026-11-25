"""Register one fixed-blend independent-content ranking follow-up."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import os,subprocess,sys
from pathlib import Path
from run_ranking_study import REPO,read,freeze
sys.path.insert(0,str(REPO))
from taac_retrieval.ranking_model import digest


def job(config,variant,seed,gpu):
    root=REPO/config['root'];folder=root/'training'/f'{variant}_seed{seed}'
    if (folder/'result.json').exists():return
    cmd=[sys.executable,'-m','taac_retrieval.content_ranker_train','--variant',variant,'--seed',str(seed),'--device',f'cuda:{gpu}']
    freeze(root/f'{variant}_seed{seed}.command.json',dict(command=cmd))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
    print(f'Started content {variant} seed{seed}',flush=True)
    with (root/f'{variant}_seed{seed}.log').open('x') as log:subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    r=read(folder/'result.json')
    print(f'Completed content {variant} seed{seed}: {r["validation"]["blended"]["HR10"]}',flush=True)


def main():
    config=read(REPO/'configs/ranking_content_study.json');root=REPO/config['root'];ref=REPO/config['reference_root']
    assert read(ref/'training_complete.json')['runs']==4
    old=read(ref/'source_manifest.json')
    for name,sha in old.items():assert digest(REPO/name)==sha,name
    names=list(old)+['configs/ranking_content_study.json','taac_retrieval/content_ranker.py',
        'taac_retrieval/content_ranker_train.py','scripts/run_content_ranking_study.py','tests/test_content_ranker.py']
    freeze(root/'plan.json',config);freeze(root/'source_manifest.json',{n:digest(REPO/n) for n in names})
    if not (root/'registered.json').exists():
        freeze(root/'registered.json',dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
            phase='Exploratory after first-stage frozen-vector ranking results',test_evaluated=False,
            first_stage_results_sha256={f'{v}_{s}':digest(ref/f'training/{v}_seed{s}/result.json') for v in config['variants'] for s in config['seeds']}))
    for seed in config['seeds']:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(job,config,v,seed,gpu) for v,gpu in [('mean_mlp',5),('din_attention',6)]]
            for f in futures:f.result()
    for name,sha in read(root/'source_manifest.json').items():assert digest(REPO/name)==sha,name
    freeze(root/'training_complete.json',dict(runs=4,test_evaluated=False,blend_alpha=config['blend']['alpha']))


if __name__=='__main__':main()
