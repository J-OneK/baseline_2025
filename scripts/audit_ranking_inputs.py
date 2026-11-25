"""Independent checks for logged-event labels, time boundaries and frozen feature caches."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from taac_retrieval.data import PreparedData,make_input
from taac_retrieval.evaluate_checkpoint import load_model
from taac_retrieval.ranking_model import digest
from run_ranking_study import read,freeze


def histories(items,actions,cuts,length):
    positions=np.flatnonzero(actions==2);values=items[positions]
    if not len(values):return np.zeros((len(cuts),length),np.int32)
    ends=np.searchsorted(positions,cuts,side='left')
    index=ends[:,None]+np.arange(-length,0)[None,:]
    return np.where(index>=0,values[np.maximum(index,0)],0)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='reports/ranking_study/input_audit.json')
    args=p.parse_args();torch.set_num_threads(4)
    c=read(REPO/'configs/ranking_study.json');root=REPO/c['root'];ex=root/'examples'
    d=PreparedData(c['cache']);settings=c['data'];rng=np.random.default_rng(settings['seed'])
    samples={part:{k:np.load(ex/f'{part}_{k}.npy',mmap_mode='r') for k in ['rows','positions','cuts','items','labels','history']}
             for part in ['train','pointval']}
    events=np.zeros(d.num_items+1,np.int64);clicks=np.zeros_like(events)
    cohort=set(np.asarray(d.eval_indices).tolist());checked=0
    for row in range(d.num_users):
        n=int(d.train_cut[row]);end=int(d.test_cut[row]);items=d.items[row,:end];actions=d.actions[row,:end];times=d.timestamps[row,:end]
        boundary=int(np.searchsorted(times,times[n//2],side='left'))
        first=np.sort(np.unique(items[:n],return_index=True)[1])
        known=first[np.isin(actions[first],[1,2])]
        stat=known[known<boundary];np.add.at(events,items[stat],1);np.add.at(clicks,items[stat],actions[stat]==2)
        possible=known[(known>=boundary)&(np.searchsorted(times,times[known],side='left')>=settings['min_history'])]
        if len(possible)>settings['train_events_per_user']:
            chosen=np.sort(rng.choice(len(possible),settings['train_events_per_user'],replace=False));possible=possible[chosen]
        for part in ['train','pointval']:
            s=samples[part];lo,hi=np.searchsorted(s['rows'],[row,row+1]);pos=s['positions'][lo:hi]
            if part=='train':expected=possible
            elif row in cohort:
                first=np.sort(np.unique(items,return_index=True)[1]);known=first[np.isin(actions[first],[1,2])]
                expected=known[(known>=d.val_cut[row])&(known<end)&
                    (np.searchsorted(times,times[known],side='left')>=settings['min_history'])]
            else:expected=np.array([],dtype=int)
            np.testing.assert_array_equal(pos,expected)
            if not len(pos):continue
            cuts=np.searchsorted(times,times[pos],side='left')
            np.testing.assert_array_equal(s['cuts'][lo:hi],cuts)
            np.testing.assert_array_equal(s['items'][lo:hi],items[pos])
            np.testing.assert_array_equal(s['labels'][lo:hi],actions[pos]==2)
            np.testing.assert_array_equal(s['history'][lo:hi],histories(items,actions,cuts,settings['click_history']))
            assert not (s['history'][lo:hi]==items[pos,None]).any()
            checked+=len(pos)
        if row and row%20000==0:print(f'Audited {row} users',flush=True)
    np.testing.assert_array_equal(events,np.load(ex/'stats_events.npy'));np.testing.assert_array_equal(clicks,np.load(ex/'stats_clicks.npy'))
    errors={}
    for seed in c['seeds']:
        f=root/f'features_seed{seed}';meta=read(f/'metadata.json')
        model,_,_=load_model(meta['checkpoint'],c['cache'],'cpu')
        errors[str(seed)]={}
        for part in ['train','pointval']:
            s=samples[part];indices=np.linspace(0,len(s['rows'])-1,32,dtype=int);batch=[]
            for i in indices:
                row=int(s['rows'][i]);cut=int(s['cuts'][i])
                seq,kinds,acts=make_input(d.items[row,:cut],d.actions[row,:cut],d.max_history)
                batch.append(dict(items=seq,kinds=kinds,actions=acts,user=row+1))
            expected=model.predict(default_collate(batch)).numpy()
            actual=np.load(f/f'{part}_queries.npy',mmap_mode='r')[indices]
            error=float(np.max(np.abs(actual-expected)));assert error<5e-5,error
            errors[str(seed)][part]=error
        with np.load(f/'validation_candidates.npz') as archive:
            a={key:archive[key] for key in archive.files}
        if a:
            np.testing.assert_array_equal(a['rows'],d.eval_indices)
            np.testing.assert_array_equal(a['targets'],d.items[d.eval_indices,d.val_pos[d.eval_indices]])
            for i,row in enumerate(a['rows']):
                cut=int(d.val_cut[row]);expected=histories(d.items[row],d.actions[row],np.array([cut]),settings['click_history'])[0]
                np.testing.assert_array_equal(expected,a['history'][i])
                assert len(np.unique(a['candidates'][i]))==500
                assert not np.intersect1d(a['candidates'][i],a['seen'][i]).size
            iv=np.load(f/'item_vectors.npy',mmap_mode='r')
            scores=np.einsum('bi,bki->bk',a['queries'],iv[a['candidates']])
            assert np.max(np.abs(scores-a['scores']))<5e-5
        del model
    freeze(REPO/args.output,dict(status='passed',events_checked=checked,all_event_indices_labels_cuts_histories_reconstructed=True,
        early_statistics_independently_rebuilt=True,training_and_validation_events_disjoint=True,
        no_unknown_labels_or_repeated_ad_targets=True,queries_match_independent_truncated_histories=errors,
        candidates_are_fixed_unique_and_history_filtered=True,validation_history_checks_all_users=True,
        test_target_evaluated=False,source_sha256=digest(Path(__file__))))
    print('Input audit complete.',flush=True)


if __name__=='__main__':main()
