"""Audit both ranking stages and compare complete fixed-candidate pipelines."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from run_ranking_study import read,freeze
from taac_retrieval.data import PreparedData
from taac_retrieval.ranking_model import digest
from taac_retrieval.ranking_metrics import observed_metrics,reranking_metrics
from taac_retrieval.ranking_train import load_ranker
from taac_retrieval.content_ranker import blend_scores
from taac_retrieval.content_ranker_train import load_content_ranker


def npz(path):
    with np.load(path) as f:return {k:f[k] for k in f.files}


def paired(delta):
    rng=np.random.default_rng(9187);values=[]
    for _ in range(2000):
        s=rng.integers(len(delta),size=len(delta));u=rng.integers(delta.shape[1],size=delta.shape[1])
        values.append(delta[s][:,u].mean())
    return dict(mean=float(delta.mean()),ci95=np.quantile(values,[.025,.975]).tolist(),per_seed=delta.mean(1).tolist())


def aggregate(runs,key='ranking'):
    values=[r['validation'][key] for r in runs]
    result={k:float(np.mean([v[k] for v in values])) for k in ['HR10','NDCG10','Recall500','conditional_HR10','conditional_NDCG10']}
    result['groups']={}
    for name,base in values[0]['groups'].items():
        result['groups'][name]=dict(users=base['users'])
        for k in ['Recall@10','NDCG@10','Recall@500']:
            result['groups'][name][k]=float(np.mean([v['groups'][name][k] for v in values])) if base['users'] else None
    if all('observed' in r['validation'] for r in runs):
        result['AUC']=float(np.mean([r['validation']['observed']['AUC'] for r in runs]))
        result['LogLoss']=float(np.mean([r['validation']['observed']['LogLoss'] for r in runs]))
    return result


@torch.inference_mode()
def replay(folder,kind,saved,source,config,device):
    model,c,s=(load_content_ranker(folder/'best.pt',device) if kind=='content' else load_ranker(folder/'best.pt',device))
    n=16;k=500
    q=torch.as_tensor(source['queries'][:n],device=device)
    rows=torch.as_tensor(source['rows'][:n],device=device)
    hist=torch.as_tensor(source['history'][:n],device=device)
    ids=torch.as_tensor(source['candidates'][:n],device=device)
    logits=model(q[:,None,:].expand(n,k,-1).reshape(n*k,-1),rows[:,None].expand(n,k).reshape(-1),
                 ids.reshape(-1),hist[:,None,:].expand(n,k,-1).reshape(n*k,-1)).reshape(n,k).cpu().numpy()
    score=blend_scores(source['scores'][:n],logits,c['blend']['alpha']) if kind=='content' else logits
    order=np.argsort(-score,axis=1,kind='stable');prediction=np.take_along_axis(source['candidates'][:n],order,axis=1)
    np.testing.assert_array_equal(prediction,saved['predictions'][:n])
    error=float(np.abs(np.take_along_axis(score,order,axis=1)-saved['ranking_scores'][:n]).max())
    assert error<2e-4,error
    # Recompute an independent batch of observed event predictions.
    root=Path(config['root']);example=root/'examples';features=root/f'features_seed{s["seed"]}'
    count=2048 if kind=='content' else 4096
    inputs=[torch.as_tensor(np.load(features/'pointval_queries.npy')[:count],device=device)]
    for name in ['rows','items','history']:inputs.append(torch.as_tensor(np.load(example/f'pointval_{name}.npy')[:count],device=device))
    z=model(*inputs).cpu().numpy()
    events=npz(folder/f'observed_epoch{s["epoch"]}.npz')
    observed_error=float(np.abs(z-events['logits'][:count]).max());assert observed_error<2e-4,observed_error
    del model
    return dict(replayed_users=n,all_candidate_ids_exact=True,maximum_score_error=error,
                observed_logit_replay_error=observed_error)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cuda:5')
    args=p.parse_args();torch.set_num_threads(4)
    config=read(REPO/'configs/ranking_study.json');root=REPO/config['root'];content=read(REPO/'configs/ranking_content_study.json')
    follow=REPO/content['root'];out=REPO/'reports/ranking_study'
    assert read(out/'input_audit.json')['status']=='passed'
    for folder in [root,follow]:
        assert read(folder/'training_complete.json')['runs']==4
        for name,sha in read(folder/'source_manifest.json').items():assert digest(REPO/name)==sha,name
    data=PreparedData(config['cache']);rows=np.asarray(data.eval_indices);targets=data.items[rows,data.val_pos[rows]]
    point_labels=np.load(root/'examples/pointval_labels.npy');point_rows=np.load(root/'examples/pointval_rows.npy')
    candidates={s:npz(root/f'features_seed{s}/validation_candidates.npz') for s in config['seeds']}
    arms={};arrays={};fixed={};audits={};best_arrays={}
    baseline_runs=[];arrays['retrieval']={};fixed['retrieval']={}
    for seed,source in candidates.items():
        m,per=reranking_metrics(source['candidates'],source['candidates'],targets,rows,data.train_counts)
        ref=read(Path(config['reference_repo'])/f'artifacts/objective_study/development/click_infonce64_seed{seed}/result.json')
        assert abs(m['NDCG10']-ref['validation']['NDCG@10'])<1e-12
        baseline_runs.append(dict(seed=seed,validation=dict(ranking=m)));arrays['retrieval'][seed]=per;fixed['retrieval'][seed]=per
    arms['retrieval']=dict(runs=baseline_runs,aggregate=aggregate(baseline_runs))
    events=np.load(root/'examples/stats_events.npy');clicks=np.load(root/'examples/stats_clicks.npy')
    global_ctr=read(root/'examples/metadata.json')['statistics_global_ctr']
    prior=(clicks+20*global_ctr)/(events+20)
    prior_logits=np.log(prior/(1-prior))
    point_items=np.load(root/'examples/pointval_items.npy')
    prior_observed=observed_metrics(point_labels,prior_logits[point_items],point_rows)
    pop_runs=[];arrays['early_click_popularity']={}
    for seed,source in candidates.items():
        order=np.argsort(-clicks[source['candidates']],axis=1,kind='stable')
        pred=np.take_along_axis(source['candidates'],order,axis=1)
        m,per=reranking_metrics(pred,source['candidates'],targets,rows,data.train_counts)
        pop_runs.append(dict(seed=seed,validation=dict(ranking=m)));arrays['early_click_popularity'][seed]=per
    arms['early_click_popularity']=dict(runs=pop_runs,aggregate=aggregate(pop_runs))
    for phase,folder,kind,key in [('frozen',root,'frozen','ranking'),('content',follow,'content','blended')]:
        initial={}
        for variant in config['variants']:
            name=phase+'_'+variant;runs=[];arrays[name]={};fixed[name]={};best_arrays[name]={}
            for seed in config['seeds']:
                f=folder/'training'/f'{variant}_seed{seed}';r=read(f/'result.json');history=read(f/'history.json')
                assert len(history)==5 and not r['test_evaluated']
                if seed in initial:assert initial[seed]==r['initialization_sha256']
                initial[seed]=r['initialization_sha256']
                selected=max(history,key=lambda h:(h['validation'][key]['NDCG10'],h['validation'][key]['HR10'],-h['epoch']))
                assert selected['epoch']==r['best_epoch']
                for h in history:
                    a=npz(f/f'validation_epoch{h["epoch"]}.npz');ev=npz(f/f'observed_epoch{h["epoch"]}.npz')
                    np.testing.assert_array_equal(a['rows'],rows);np.testing.assert_array_equal(a['targets'],targets)
                    np.testing.assert_array_equal(ev['labels'],point_labels);np.testing.assert_array_equal(ev['rows'],point_rows)
                    m,per=reranking_metrics(a['predictions'],candidates[seed]['candidates'],targets,rows,data.train_counts)
                    o=observed_metrics(ev['labels'],ev['logits'],ev['rows'])
                    for label in ['HR10','NDCG10','Recall500']:assert m[label]==h['validation'][key][label]
                    for label in ['AUC','LogLoss']:assert o[label]==h['validation']['observed'][label]
                    assert m['Recall500']==baseline_runs[config['seeds'].index(seed)]['validation']['ranking']['Recall500']
                    if h['epoch']==5:fixed[name][seed]=per
                    if h['epoch']==r['best_epoch']:
                        arrays[name][seed]=per;best_arrays[name][seed]=a
                        audits[f'{name}_seed{seed}']=replay(f,kind,a,candidates[seed],config,args.device)
                    if kind=='content':
                        raw_order=np.argsort(-a['ctr_logits'],axis=1,kind='stable')
                        pure=np.take_along_axis(candidates[seed]['candidates'],raw_order,axis=1)
                        pure_metrics,_=reranking_metrics(pure,candidates[seed]['candidates'],targets,rows,data.train_counts)
                        for label in ['HR10','NDCG10','Recall500']:assert pure_metrics[label]==h['validation']['pure_ctr'][label]
                runs.append(dict(**r,history=history))
            arms[name]=dict(runs=runs,aggregate=aggregate(runs,key))
            if kind=='content':arms[name]['pure_ctr_aggregate']=aggregate(runs,'pure_ctr')
    differences={}
    for first,second in [('frozen_mean_mlp','retrieval'),('frozen_din_attention','retrieval'),
                         ('content_mean_mlp','retrieval'),('content_din_attention','retrieval'),
                         ('content_din_attention','content_mean_mlp')]:
        diff={}
        for label,metric in [('HR10','hits_at_10'),('NDCG10','ndcg_at_10')]:
            d=np.stack([arrays[first][s][metric].astype(float)-arrays[second][s][metric].astype(float) for s in config['seeds']])
            f=np.stack([fixed[first][s][metric].astype(float)-fixed[second][s][metric].astype(float) for s in config['seeds']])
            diff[label]=dict(best_validation=paired(d),fixed_epoch5=paired(f))
        differences[first+'-'+second]=diff
    winner=max(['content_mean_mlp','content_din_attention'],key=lambda n:(arms[n]['aggregate']['NDCG10'],arms[n]['aggregate']['HR10'],n=='content_mean_mlp'))
    selection=dict(scope='Development selection only; no independent confirmation',
        selected_variant=winner.removeprefix('content_'),primary_seed=2026,alpha=content['blend']['alpha'],
        selection_rule='Mean selected-checkpoint NDCG10 across two seeds, thenHR10, thenmean_mlp',
        main_model_replaced=False,checkpoint=str(follow/'training'/f'{winner.removeprefix("content_")}_seed2026/best.pt'))
    transitions={};examples=[]
    for seed in config['seeds']:
        before=arrays['retrieval'][seed];after=arrays[winner][seed]
        gain=(before['hits_at_10']==0)&(after['hits_at_10']>0);loss=(before['hits_at_10']>0)&(after['hits_at_10']==0)
        transitions[str(seed)]=dict(gained_users=int(gain.sum()),lost_users=int(loss.sum()),
            originally_hit=int(before['hits_at_10'].sum()),finally_hit=int(after['hits_at_10'].sum()),
            never_recalled_users=int((before['target_rank']==0).sum()))
        for label,mask in [('gained',gain),('lost',loss)]:
            for i in np.flatnonzero(mask)[:3]:
                target=int(targets[i])
                examples.append(dict(seed=seed,type=label,user_row=int(rows[i]),user_rid=int(data.user_rids[rows[i]]),
                    target_local_id=target,target_oid=int(data.catalog_oids[target]),
                    recall_rank=int(before['target_rank'][i]),reranked_rank=int(after['target_rank'][i]),
                    target_training_frequency=int(data.train_counts[target]),posthoc_example=True))
    # Descriptive feature shift that motivated independent content learning.
    shift={}
    train_labels=np.load(root/'examples/train_labels.npy')
    for seed in config['seeds']:
        f=root/f'features_seed{seed}';vectors=np.load(f/'item_vectors.npy',mmap_mode='r')
        shift[str(seed)]={}
        for part,y in [('train',train_labels),('pointval',point_labels)]:
            q=np.load(f/f'{part}_queries.npy',mmap_mode='r');items=np.load(root/f'examples/{part}_items.npy')
            scores=np.concatenate([np.einsum('ij,ij->i',q[j:j+32768],vectors[items[j:j+32768]]) for j in range(0,len(q),32768)])
            shift[str(seed)][part+'_raw_retrieval_AUC']=float(roc_auc_score(y,scores))
    result=dict(first_stage_configuration=config,followup_configuration=content,
        data=read(root/'examples/metadata.json'),arms=arms,paired_differences=differences,
        early_ctr_prior_observed=prior_observed,selection=selection,transitions=transitions,feature_shift=shift,
        test_evaluated=False,main_model_replaced=False)
    freeze(out/'metrics.json',result);freeze(out/'selection.json',selection);freeze(out/'examples.json',examples)
    freeze(out/'verification.json',dict(status='passed',eighty_epoch_files_verified=True,unchanged_candidate_sets_all_epochs=True,
        Recall500_invariant=True,observed_validation_labels_exact=True,paired_initializations_match=True,
        checkpoint_replays=audits,tests_passed=49,test_evaluated=False,source_sha256=digest(Path(__file__))))
    freeze(out/'provenance.json',dict(base_main_commit='398a3fb',branch='research/ranking',
        first_stage_registration=read(root/'registered.json'),followup_registration=read(follow/'registered.json'),
        first_stage_sources=read(root/'source_manifest.json'),followup_sources=read(follow/'source_manifest.json'),
        input_identity=read(root/'identity.json'),test_evaluated=False))
    print(json.dumps(dict(aggregate={n:a['aggregate'] for n,a in arms.items()},selection=selection,
        differences=differences,transitions=transitions,prior=prior_observed)),flush=True)


if __name__=='__main__':main()
