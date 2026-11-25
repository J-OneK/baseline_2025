"""One-shot new-cohort confirmation after freezing ranking architecture and epochs."""
import argparse,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from run_ranking_study import read,freeze
from finish_ranking_study import paired
from taac_retrieval.data import PreparedData,SequenceDataset
from taac_retrieval.evaluate_checkpoint import load_model
from taac_retrieval.metrics import to_device,exact_topk
from taac_retrieval.ranking_data import history_clicks
from taac_retrieval.ranking_model import digest
from taac_retrieval.ranking_metrics import reranking_metrics
from taac_retrieval.content_ranker import blend_scores
from taac_retrieval.content_ranker_train import load_content_ranker

ROOT=REPO/'artifacts/ranking_confirmation_v1'


def register():
    selection=read(REPO/'reports/ranking_study/selection.json')
    assert selection['selected_variant']=='mean_mlp' and selection['alpha']==.1
    base=read(REPO/'configs/ranking_study.json');data=PreparedData(base['cache'])
    sources={};excluded=set();missing=[]
    def add(indices,userfile):
        indices,userfile=Path(indices).resolve(),Path(userfile).resolve()
        if not indices.exists() or not userfile.exists():missing.append([str(indices),str(userfile)]);return
        key=str(indices)+'|'+str(userfile)
        if key in sources:return
        rows=np.load(indices,allow_pickle=False);users=np.load(userfile,mmap_mode='r',allow_pickle=False)
        if rows.ndim!=1 or len(rows)==0 or not np.issubdtype(rows.dtype,np.integer) or rows.min()<0 or rows.max()>=len(users):
            raise ValueError(f'Unrecognised cohort indexing: {indices}')
        values=np.asarray(users[rows]);excluded.update(values.tolist())
        sources[key]=dict(indices=str(indices),indices_sha256=digest(indices),user_rids=str(userfile),
                          user_rids_sha256=digest(userfile),users=len(rows))
    for repo in sorted(REPO.parent.glob('baseline_2025*')):
        artifacts=repo/'artifacts'
        if not artifacts.exists():continue
        for path in artifacts.rglob('eval_indices.npy'):
            if (path.parent/'user_rids.npy').exists():add(path,path.parent/'user_rids.npy')
        for path in artifacts.rglob('config.json'):
            try:c=read(path)
            except (ValueError,OSError):continue
            if c.get('eval_indices') and c.get('cache'):
                cache=Path(c['cache']);cache=cache if cache.is_absolute() else repo/cache
                indices=Path(c['eval_indices']);indices=indices if indices.is_absolute() else repo/indices
                add(indices,cache/'user_rids.npy')
        for identity in artifacts.glob('*/cohorts/identity.json'):
            c=read(identity)
            if 'cache' in c:
                for indices in identity.parent.glob('*.npy'):add(indices,Path(c['cache'])/'user_rids.npy')
            for s in c.get('sources',[]):
                userfile=s.get('user_rids',s.get('user_rids_file'))
                if userfile:add(s['indices'],userfile)
    assert not missing,missing
    eligible=np.flatnonzero(~np.isin(data.user_rids,np.array(sorted(excluded),dtype=np.int64)))
    rng=np.random.default_rng(5481);chosen=np.sort(rng.choice(eligible,5000,replace=False))
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'cohort.npy'
    if path.exists():np.testing.assert_array_equal(np.load(path),chosen)
    else:np.save(path,chosen)
    assert not set(np.asarray(data.user_rids)[chosen].tolist())&excluded
    refs={}
    for seed in base['seeds']:
        checkpoint=REPO/f'artifacts/ranking_content_v1/training/mean_mlp_seed{seed}/best.pt'
        refs[str(seed)]=dict(path=str(checkpoint),sha256=digest(checkpoint))
    freeze(ROOT/'plan.json',dict(seed=5481,users=5000,variant='mean_mlp',alpha=.1,
        selected_checkpoints=refs,selection_file_sha256=digest(REPO/'reports/ranking_study/selection.json'),
        split='val',scope='New held-out events of training-population users, not new-user cold start',
        rule='One-shot confirmation, no epoch/architecture/alpha changes after results',test_evaluated=False))
    freeze(ROOT/'cohort_identity.json',dict(exclusion_sources=list(sources.values()),excluded_unique_rids=len(excluded),
        eligible_current_users=len(eligible),cohort_sha256=digest(path),overlap_with_known_evaluation_users=0,
        selected_user_rids_sha256=__import__('hashlib').sha256(np.asarray(data.user_rids)[chosen].tobytes()).hexdigest()))
    files=list(read(REPO/'artifacts/ranking_content_v1/source_manifest.json'))+['scripts/confirm_ranking_cohort.py']
    freeze(ROOT/'source_manifest.json',{p:digest(REPO/p) for p in files})
    if not (ROOT/'registered.json').exists():freeze(ROOT/'registered.json',dict(registered_at_utc=datetime.now(timezone.utc).isoformat()))
    print(f'Frozen5000-user confirmation; excluded {len(excluded)} known cohort user IDs.',flush=True)


@torch.inference_mode()
def worker(seed,device):
    torch.set_num_threads(4);plan=read(ROOT/'plan.json');base=read(REPO/'configs/ranking_study.json')
    cp=plan['selected_checkpoints'][str(seed)];assert digest(cp['path'])==cp['sha256']
    ranker,config,_=load_content_ranker(cp['path'],device)
    feature=REPO/base['root']/f'features_seed{seed}';meta=read(feature/'metadata.json')
    retrieval,data,_=load_model(meta['checkpoint'],base['cache'],device);data.set_eval_indices(np.load(ROOT/'cohort.npy'))
    vectors=torch.as_tensor(np.load(feature/'item_vectors.npy'),device=device)
    saved={k:[] for k in ['candidates','predictions','rows','targets']}
    for cpu in DataLoader(SequenceDataset(data,'val'),batch_size=128,shuffle=False):
        b=to_device(cpu,device);q=retrieval.predict(b)
        candidate=exact_topk(q,vectors,b['items'],500);recall=(q[:,None,:]*vectors[candidate]).sum(-1).cpu().numpy()
        histories=np.stack([history_clicks(data.items[r],data.actions[r],int(data.val_cut[r]),20) for r in cpu['row'].numpy()])
        # Small candidate batches match deployment inference and bound peak memory.
        logits=[]
        for start in range(0,len(q),16):
            n=min(16,len(q)-start);k=500;h=torch.as_tensor(histories[start:start+n],device=device)
            z=ranker(q[start:start+n,None,:].expand(n,k,-1).reshape(n*k,-1),
                b['row'][start:start+n,None].expand(n,k).reshape(-1),candidate[start:start+n].reshape(-1),
                h[:,None,:].expand(n,k,-1).reshape(n*k,-1))
            logits.append(z.reshape(n,k).cpu().numpy())
        logits=np.concatenate(logits);score=blend_scores(recall,logits,.1);order=np.argsort(-score,axis=1,kind='stable')
        cand=candidate.cpu().numpy();ranked=np.take_along_axis(cand,order,axis=1)
        assert not (b['items']==b['target'][:,None]).any()
        saved['candidates'].append(cand);saved['predictions'].append(ranked)
        saved['rows'].append(cpu['row'].numpy());saved['targets'].append(cpu['target'].numpy())
    saved={k:np.concatenate(v) for k,v in saved.items()}
    baseline,before=reranking_metrics(saved['candidates'],saved['candidates'],saved['targets'],saved['rows'],data.train_counts)
    ranked,after=reranking_metrics(saved['predictions'],saved['candidates'],saved['targets'],saved['rows'],data.train_counts)
    assert baseline['Recall500']==ranked['Recall500']
    np.savez_compressed(ROOT/f'seed{seed}.npz',**saved,**{'baseline_'+k:v for k,v in before.items()},**{'ranked_'+k:v for k,v in after.items()})
    freeze(ROOT/f'seed{seed}.json',dict(seed=seed,baseline=baseline,ranked=ranked,checkpoint_sha256=cp['sha256'],
        gained_users=int(((before['hits_at_10']==0)&(after['hits_at_10']>0)).sum()),
        lost_users=int(((before['hits_at_10']>0)&(after['hits_at_10']==0)).sum()),test_evaluated=False))
    print(json.dumps(dict(seed=seed,baseline=baseline['HR10'],ranked=ranked['HR10'])),flush=True)


def run(seed,gpu):
    if (ROOT/f'seed{seed}.json').exists():return
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
    with (ROOT/f'seed{seed}.log').open('x') as log:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker','--seed',str(seed),'--device',f'cuda:{gpu}'],
            cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true')
    p.add_argument('--seed',type=int);p.add_argument('--device',default='cuda:5');a=p.parse_args()
    if a.worker:worker(a.seed,a.device);return
    register()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,s,g) for s,g in [(2026,5),(2027,6)]]
        for f in futures:f.result()
    runs=[read(ROOT/f'seed{s}.json') for s in [2026,2027]];differences={}
    for name,key in [('HR10','hits_at_10'),('NDCG10','ndcg_at_10')]:
        delta=[]
        for s in [2026,2027]:
            with np.load(ROOT/f'seed{s}.npz') as d:delta.append(d['ranked_'+key].astype(float)-d['baseline_'+key].astype(float))
        differences[name]=paired(np.stack(delta))
    aggregate={arm:{k:float(np.mean([r[arm][k] for r in runs])) for k in ['HR10','NDCG10','Recall500']} for arm in ['baseline','ranked']}
    for name,sha in read(ROOT/'source_manifest.json').items():assert digest(REPO/name)==sha,name
    result=dict(plan=read(ROOT/'plan.json'),registration=read(ROOT/'registered.json'),
        cohort_identity=read(ROOT/'cohort_identity.json'),runs=runs,aggregate=aggregate,paired_differences=differences,
        test_evaluated=False,no_post_confirmation_tuning=True,source_sha256=digest(Path(__file__)))
    freeze(REPO/'reports/ranking_study/confirmation.json',result)
    print(json.dumps(dict(aggregate=aggregate,differences=differences)),flush=True)


if __name__=='__main__':main()
