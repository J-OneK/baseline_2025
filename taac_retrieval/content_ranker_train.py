"""Controlled follow-up: independent CTR ranker plus one fixed recall/CTR blend."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import PreparedData
from .ranking_data import read,write_new
from .ranking_train import tensors
from .ranking_model import digest
from .ranking_metrics import observed_metrics,reranking_metrics
from .content_ranker import ContentRanker,blend_scores
from .train import seed_everything,initialization_hash


@torch.inference_mode()
def evaluate_content(model,config,seed,device,point=None):
    model.eval();base=read(config['base'])
    point=tensors(base,seed,'pointval',device) if point is None else point
    point_logits=[]
    for start in range(0,len(point['rows']),2048):
        s=slice(start,start+2048)
        point_logits.append(model(point['queries'][s],point['rows'][s],point['items'][s],point['history'][s]).cpu().numpy())
    point_logits=np.concatenate(point_logits)
    observed=observed_metrics(point['labels'].cpu().numpy(),point_logits,point['rows'].cpu().numpy())
    with np.load(Path(config['reference_root'])/f'features_seed{seed}/validation_candidates.npz') as source:
        data={k:source[k] for k in source.files}
    logits=[];k=data['candidates'].shape[1]
    for start in range(0,len(data['rows']),16):
        rows=torch.as_tensor(data['rows'][start:start+16],device=device)
        q=torch.as_tensor(data['queries'][start:start+16],device=device)
        h=torch.as_tensor(data['history'][start:start+16],device=device)
        items=torch.as_tensor(data['candidates'][start:start+16],device=device);b=len(rows)
        z=model(q[:,None,:].expand(b,k,-1).reshape(b*k,-1),rows[:,None].expand(b,k).reshape(-1),
                items.reshape(-1),h[:,None,:].expand(b,k,-1).reshape(b*k,-1))
        logits.append(z.reshape(b,k).cpu().numpy())
    logits=np.concatenate(logits);counts=PreparedData(base['cache']).train_counts
    metrics={};arrays={}
    for name,scores in [('blended',blend_scores(data['scores'],logits,config['blend']['alpha'])),('pure_ctr',logits)]:
        order=np.argsort(-scores,axis=1,kind='stable')
        found=np.take_along_axis(data['candidates'],order,axis=1)
        m,per=reranking_metrics(found,data['candidates'],data['targets'],data['rows'],counts)
        metrics[name]=m
        if name=='blended':
            arrays=dict(predictions=found,ranking_scores=np.take_along_axis(scores,order,axis=1),
                        rows=data['rows'],targets=data['targets'],ctr_logits=logits,**per)
    metrics['observed']=observed
    return metrics,arrays,dict(logits=point_logits,labels=point['labels'].cpu().numpy(),rows=point['rows'].cpu().numpy())


def load_content_ranker(checkpoint,device):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);config=saved['configuration']
    assert digest(Path(read(config['base'])['cache'])/'metadata.json')==saved['cache_metadata_sha256']
    model=ContentRanker(config,saved['variant']);model.load_state_dict(saved['model'])
    return model.to(device).eval(),config,saved


def train(args):
    config=read(args.config);base=read(config['base']);root=Path(config['root']);t=config['training']
    out=root/'training'/f'{args.variant}_seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    if (out/'config.json').exists():raise FileExistsError('Incomplete run; choose a new root.')
    seed_everything(args.seed);model=ContentRanker(config,args.variant);initial=initialization_hash(model)
    info=dict(configuration=config,seed=args.seed,variant=args.variant,initialization_sha256=initial,
        parameters=sum(p.numel() for p in model.parameters()),embedded_fields=model.embedded_fields,
        test_evaluated=False)
    write_new(out/'config.json',info);model=model.to(args.device)
    train=tensors(base,args.seed,'train',args.device);point=tensors(base,args.seed,'pointval',args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
    rng=torch.Generator(device=args.device).manual_seed(args.seed+111);best=(-1.,-1.);history=[]
    for epoch in range(1,t['epochs']+1):
        model.train();start=time.perf_counter();loss_sum=0.
        order=torch.randperm(len(train['rows']),device=args.device,generator=rng)
        for indices in order.split(t['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            z=model(train['queries'][indices],train['rows'][indices],train['items'][indices],train['history'][indices])
            loss=F.binary_cross_entropy_with_logits(z,train['labels'][indices].float())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite content ranker loss.')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),t['grad_clip'],error_if_nonfinite=True)
            optimizer.step();loss_sum+=float(loss.detach())*len(indices)
        seconds=time.perf_counter()-start
        metrics,arrays,observed=evaluate_content(model,config,args.seed,args.device,point)
        np.savez_compressed(out/f'validation_epoch{epoch}.npz',**arrays)
        np.savez_compressed(out/f'observed_epoch{epoch}.npz',**observed)
        row=dict(epoch=epoch,training_loss=loss_sum/len(order),training_seconds=seconds,validation=metrics)
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps(dict(event='epoch',seed=args.seed,variant=args.variant,**row)),flush=True)
        score=metrics['blended']['NDCG10'],metrics['blended']['HR10']
        if score>best:
            best=score
            torch.save(dict(model=model.state_dict(),epoch=epoch,configuration=config,seed=args.seed,variant=args.variant,
                validation=metrics,cache_metadata_sha256=digest(Path(base['cache'])/'metadata.json')),out/'best.pt')
    chosen=max(history,key=lambda h:(h['validation']['blended']['NDCG10'],h['validation']['blended']['HR10'],-h['epoch']))
    write_new(out/'result.json',dict(seed=args.seed,variant=args.variant,best_epoch=chosen['epoch'],
        validation=chosen['validation'],fixed_epoch5=history[-1]['validation'],initialization_sha256=initial,
        parameters=info['parameters'],training_seconds=sum(h['training_seconds'] for h in history),
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated(args.device)/1024**2,
        checkpoint_sha256=digest(out/'best.pt'),test_evaluated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/ranking_content_study.json')
    p.add_argument('--variant',choices=['mean_mlp','din_attention'],required=True)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--device',default='cuda:5')
    train(p.parse_args())
