"""Demonstrate retrieval Top500 -> learned content ranking -> final Top10."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .data import PreparedData
from .data import SequenceDataset
from .evaluate_checkpoint import load_model
from .metrics import exact_topk,to_device
from torch.utils.data import default_collate
from .ranking_data import read
from .content_ranker import blend_scores
from .content_ranker_train import load_content_ranker


@torch.inference_mode()
def recommend(args):
    torch.set_num_threads(4)
    checkpoint=args.checkpoint or read('reports/ranking_study/selection.json')['checkpoint']
    model,config,saved=load_content_ranker(checkpoint,args.device)
    base=read(config['base']);data=PreparedData(base['cache']);seed=saved['seed']
    folder=Path(config['reference_root'])/f'features_seed{seed}'
    if args.cached:
        with np.load(folder/'validation_candidates.npz') as f:
            record={k:f[k][args.index:args.index+1] for k in f.files}
    else:
        meta=read(folder/'metadata.json')
        retrieval,data,_=load_model(meta['checkpoint'],base['cache'],args.device)
        sample=SequenceDataset(data,'val')[args.index]
        batch=to_device(default_collate([sample]),args.device)
        query=retrieval.predict(batch)
        vectors=torch.as_tensor(np.load(folder/'item_vectors.npy'),device=args.device)
        found=exact_topk(query,vectors,batch['items'],k=500)
        scores=(query[:,None,:]*vectors[found]).sum(-1)
        row=sample['row'];cut=int(data.val_cut[row])
        clicks=data.items[row,:cut][data.actions[row,:cut]==2][-base['data']['click_history']:]
        history=np.pad(clicks,(base['data']['click_history']-len(clicks),0))
        record=dict(rows=np.array([row]),targets=np.array([sample['target']]),
            candidates=found.cpu().numpy(),scores=scores.cpu().numpy(),queries=query.cpu().numpy(),history=history[None])
        del retrieval,vectors
    if not len(record['rows']):raise IndexError('Evaluation user index outside cohort.')
    ids=torch.as_tensor(record['candidates'][0],device=args.device);n=len(ids)
    row=int(record['rows'][0]);hist=torch.as_tensor(record['history'],device=args.device)
    q=torch.as_tensor(record['queries'],device=args.device)
    logits,attention=model(q.expand(n,-1),torch.full((n,),row,device=args.device,dtype=torch.long),
                            ids,hist.expand(n,-1),return_attention=True)
    ctr=logits.cpu().numpy()[None]
    scores=blend_scores(record['scores'],ctr,config['blend']['alpha'])[0]
    order=np.argsort(-scores,kind='stable')[:args.k]
    recommendations=[]
    for rank,j in enumerate(order,1):
        item=int(record['candidates'][0,j]);weights=attention[j].cpu().numpy()
        h=record['history'][0];positions=np.argsort(-weights)
        related=[dict(local_id=int(h[p]),oid=int(data.catalog_oids[h[p]]),attention_weight=float(weights[p]))
                 for p in positions if h[p]>0][:3]
        recommendations.append(dict(rank=rank,local_id=item,oid=int(data.catalog_oids[item]),recall_rank=int(j+1),
            blended_score=float(scores[j]),observed_event_click_score=float(torch.sigmoid(logits[j])),
            history_attention=related))
    result=dict(split='val',evaluation_index=args.index,user_row=row,user_rid=int(data.user_rids[row]),
        retrieval_execution='Cached fixed candidates' if args.cached else 'Fresh causal query and exactTop500 over frozen item vectors',
        seed=seed,variant=saved['variant'],checkpoint_epoch=saved['epoch'],retrieved_candidates=n,
        blend_alpha=config['blend']['alpha'],
        score_scope='CTR score is for the selected logged-event cohort; blended score is a ranking score, not CTR.',
        history_aggregation='Candidate-conditioned softmax' if saved['variant']=='din_attention' else 'Masked mean; listed weights are uniform',
        recommendations=recommendations)
    if args.show_target:
        target=int(record['targets'][0]);positions=np.flatnonzero(record['candidates'][0]==target)
        all_order=np.argsort(-scores,kind='stable')
        result['post_ranking_audit']=dict(target_local_id=target,target_oid=int(data.catalog_oids[target]),
            recall_rank=int(positions[0]+1) if len(positions) else None,
            reranked_rank=int(np.flatnonzero(all_order==positions[0])[0]+1) if len(positions) else None,
            note='Target identity is used for display only after scoring; never injected into candidates.')
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x') as f:f.write(text)
    print(text,end='')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint')
    p.add_argument('--index',type=int,default=0);p.add_argument('--k',type=int,default=10);p.add_argument('--device',default='cuda:5')
    p.add_argument('--show-target',action='store_true');p.add_argument('--output')
    p.add_argument('--cached',action='store_true',help='Replay cached candidates instead of recomputing exact retrieval.')
    args=p.parse_args()
    if not 1<=args.k<=500:p.error('k must be within1..500')
    recommend(args)
