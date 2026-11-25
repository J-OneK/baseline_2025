"""Causal logged-event samples and frozen retrieval features for ranking."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .data import PreparedData,SequenceDataset,make_input
from .evaluate_checkpoint import load_model
from .metrics import to_device,exact_topk
from .train import seed_everything


def read(path):return json.loads(Path(path).read_text())


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def eligible_events(items,actions,times,train_cut,test_cut,val_cut,settings):
    """Counts and examples never include the current target in its own history."""
    stats_cut=int(np.searchsorted(times,times[train_cut//2],side='left'))
    seen=set();stats=[];train=[];validation=[]
    for pos in range(test_cut):
        item=int(items[pos]);action=int(actions[pos])
        cut=int(np.searchsorted(times,times[pos],side='left'))
        first=item not in seen
        if first and action in (1,2):
            if pos<stats_cut:stats.append(pos)
            elif pos<train_cut and cut>=settings['min_history']:train.append((pos,cut))
            if pos>=val_cut and cut>=settings['min_history']:validation.append((pos,cut))
        seen.add(item)
    return stats_cut,stats,train,validation


def history_clicks(items,actions,cut,length):
    chosen=np.asarray(items[:cut])[np.asarray(actions[:cut])==2][-length:]
    return np.pad(chosen,(length-len(chosen),0)).astype(np.int32)


def build_examples(config):
    out=Path(config['root'])/'examples';out.mkdir(parents=True,exist_ok=True)
    if (out/'metadata.json').exists():return
    data=PreparedData(config['cache']);s=config['data'];rng=np.random.default_rng(s['seed'])
    if s['statistics_fraction']!=.5:raise ValueError('This study freezes the early-half statistic period.')
    event_counts=np.zeros(data.num_items+1,np.int64);click_counts=np.zeros_like(event_counts)
    arrays={part:{k:[] for k in ['rows','positions','cuts','items','labels','history']} for part in ['train','pointval']}
    evaluation=set(np.asarray(data.eval_indices).tolist());stats_cuts=np.zeros(data.num_users,np.int32)
    def append(part,row,pos,cut):
        arrays[part]['rows'].append(row);arrays[part]['positions'].append(pos);arrays[part]['cuts'].append(cut)
        arrays[part]['items'].append(int(data.items[row,pos]));arrays[part]['labels'].append(int(data.actions[row,pos]==2))
        arrays[part]['history'].append(history_clicks(data.items[row],data.actions[row],cut,s['click_history']))
    for row in range(data.num_users):
        n=int(data.lengths[row]);items=data.items[row,:n];actions=data.actions[row,:n];times=data.timestamps[row,:n]
        train_cut=int(data.train_cut[row]);test_cut=int(data.test_cut[row])
        stats_cut,stats,eligible,val=eligible_events(items,actions,times,train_cut,test_cut,int(data.val_cut[row]),s)
        stats_cuts[row]=stats_cut
        for pos in stats:event_counts[items[pos]]+=1;click_counts[items[pos]]+=actions[pos]==2
        if len(eligible)>s['train_events_per_user']:
            selected=np.sort(rng.choice(len(eligible),s['train_events_per_user'],replace=False))
            eligible=[eligible[j] for j in selected]
        for pos,cut in eligible:append('train',row,pos,cut)
        if row in evaluation:
            for pos,cut in val:append('pointval',row,pos,cut)
    for name,value in [('stats_events',event_counts),('stats_clicks',click_counts),('statistics_cut',stats_cuts)]:
        np.save(out/f'{name}.npy',value,allow_pickle=False)
    summary={}
    for part,columns in arrays.items():
        for key,value in columns.items():
            dtype=np.uint8 if key=='labels' else np.int32
            np.save(out/f'{part}_{key}.npy',np.asarray(value,dtype=dtype),allow_pickle=False)
        summary[part]=dict(events=len(columns['labels']),clicks=int(sum(columns['labels'])),
                          click_rate=float(np.mean(columns['labels'])),users=len(set(columns['rows'])))
    write_new(out/'metadata.json',dict(configuration=s,summary=summary,
        statistics_events=int(event_counts.sum()),statistics_clicks=int(click_counts.sum()),
        statistics_global_ctr=float(click_counts.sum()/event_counts.sum()),
        test_target_evaluated=False,training_uses_no_point_validation_events=True))


class UserContexts(Dataset):
    def __init__(self,data,split):self.data,self.split=data,split
    def __len__(self):return self.data.num_users
    def __getitem__(self,row):
        # end-1 suffices to score every event strictly before end.
        end=int(getattr(self.data,'train_cut' if self.split=='train' else 'test_cut')[row])
        seq,kinds,actions=make_input(self.data.items[row,:end-1],self.data.actions[row,:end-1],self.data.max_history)
        return dict(row=row,user=row+1,items=seq,kinds=kinds,actions=actions,start=self.data.max_history+1-end)


@torch.inference_mode()
def gather_queries(model,data,example_dir,part,device):
    rows=np.load(example_dir/f'{part}_rows.npy');cuts=np.load(example_dir/f'{part}_cuts.npy')
    result=np.empty((len(rows),model.hidden),np.float32)
    # Only affected validation users need a forward pass for point validation.
    dataset=UserContexts(data,'train' if part=='train' else 'val')
    wanted=np.unique(rows);subset=torch.utils.data.Subset(dataset,wanted.tolist())
    for cpu in DataLoader(subset,batch_size=128,shuffle=False,num_workers=0):
        b=to_device(cpu,device)
        encoded=model.encode_history(b['items'],b['kinds'],b['actions'],b['user'])
        for j,row in enumerate(cpu['row'].numpy()):
            lo,hi=np.searchsorted(rows,[row,row+1])
            locations=torch.as_tensor(cuts[lo:hi],device=device)+b['start'][j]
            result[lo:hi]=encoded[j,locations].cpu().numpy()
    assert np.isfinite(result).all()
    return result


@torch.inference_mode()
def prepare_seed(config,seed,device):
    from .ranking_model import digest
    seed_everything(seed);root=Path(config['root']);out=root/f'features_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'metadata.json').exists():return
    ref=Path(config['reference_repo'])/f'artifacts/objective_study/development/click_infonce64_seed{seed}/best.pt'
    model,data,saved=load_model(ref,config['cache'],device);example_dir=root/'examples'
    started=time.perf_counter()
    vectors=np.empty((data.num_items+1,model.hidden),np.float32)
    for start in range(0,len(vectors),8192):
        ids=torch.arange(start,min(start+8192,len(vectors)),device=device)
        vectors[start:start+len(ids)]=model.encode_items(ids).cpu().numpy()
    users=np.empty((data.num_users+1,model.hidden),np.float32)
    for start in range(0,len(users),8192):
        ids=torch.arange(start,min(start+8192,len(users)),device=device)
        users[start:start+len(ids)]=model.encode_users(ids).cpu().numpy()
    np.save(out/'item_vectors.npy',vectors);np.save(out/'user_vectors.npy',users)
    for part in ['train','pointval']:
        q=gather_queries(model,data,example_dir,part,device);np.save(out/f'{part}_queries.npy',q)
        if part=='train':
            ids=np.load(example_dir/'train_items.npy')
            raw=np.einsum('ij,ij->i',q,vectors[ids]);scale=dict(mean=float(raw.mean()),std=float(raw.std()))
    # Candidate generation is label-independent. Targets are stored only after retrieval.
    gpu_vectors=torch.as_tensor(vectors,device=device)
    arrays={k:[] for k in ['queries','rows','targets','candidates','history','seen','scores']}
    cfg=config['retrieval']
    for cpu in DataLoader(SequenceDataset(data,'val'),batch_size=cfg['batch_size'],shuffle=False):
        b=to_device(cpu,device);q=model.predict(b)
        found=exact_topk(q,gpu_vectors,b['items'],cfg['k'],cfg['candidate_chunk'])
        score=(q[:,None,:]*gpu_vectors[found]).sum(-1)
        for name,value in [('queries',q),('rows',b['row']),('targets',b['target']),('candidates',found),
                           ('seen',b['items']),('scores',score)]:
            arrays[name].append(value.cpu().numpy())
        arrays['history'].append(np.stack([history_clicks(data.items[row],data.actions[row],int(data.val_cut[row]),
                                               config['data']['click_history']) for row in cpu['row'].numpy()]))
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    assert not (arrays['seen']==arrays['targets'][:,None]).any()
    np.savez_compressed(out/'validation_candidates.npz',**arrays)
    write_new(out/'metadata.json',dict(seed=seed,checkpoint=str(ref),checkpoint_sha256=digest(ref),
        cache_metadata_sha256=digest(Path(config['cache'])/'metadata.json'),
        examples_metadata_sha256=digest(example_dir/'metadata.json'),
        raw_score_normalization=scale,preparation_seconds=time.perf_counter()-started,
        frozen_retrieval=True,selection='Original validation-selected clickInfoNCE64 checkpoint',
        files={p.name:digest(p) for p in out.glob('*.npy')},
        candidates_sha256=digest(out/'validation_candidates.npz'),test_evaluated=False))
    print(json.dumps(dict(event='features_complete',seed=seed,seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/ranking_study.json')
    p.add_argument('--action',choices=['examples','features'],required=True);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--device',default='cuda:5');args=p.parse_args();config=read(args.config)
    if args.action=='examples':build_examples(config)
    else:prepare_seed(config,args.seed,args.device)
