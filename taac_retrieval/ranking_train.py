"""Train observed-event ranking heads; evaluate only permutations of fixed candidates."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import PreparedData
from .ranking_data import read,write_new
from .ranking_model import FrozenFeatures,RankingHead,digest
from .ranking_metrics import observed_metrics,reranking_metrics
from .train import seed_everything,initialization_hash


def tensors(config,seed,part,device):
    root=Path(config['root']);ex=root/'examples';f=root/f'features_seed{seed}'
    values={'queries':np.load(f/f'{part}_queries.npy')}
    for name in ['rows','items','history','labels']:values[name]=np.load(ex/f'{part}_{name}.npy')
    return {k:torch.as_tensor(v,device=device) for k,v in values.items()}


def model_for(config,seed,variant):
    prior=read(Path(config['root'])/'examples/metadata.json')['summary']['train']['click_rate']
    return RankingHead(FrozenFeatures.load(config,seed),variant,config['model']['hidden'],config['model']['dropout'],prior)


@torch.inference_mode()
def evaluate(model,config,seed,device,point=None):
    model.eval();root=Path(config['root'])
    point=tensors(config,seed,'pointval',device) if point is None else point
    logits=[]
    for start in range(0,len(point['rows']),4096):
        s=slice(start,start+4096)
        z=model(point['queries'][s],point['rows'][s],point['items'][s],point['history'][s])
        logits.append(z.cpu().numpy())
    logits=np.concatenate(logits)
    classification=observed_metrics(point['labels'].cpu().numpy(),logits,point['rows'].cpu().numpy())
    with np.load(root/f'features_seed{seed}/validation_candidates.npz') as source:
        saved={k:source[k] for k in source.files}
    scores=[];k=saved['candidates'].shape[1]
    for start in range(0,len(saved['rows']),16):
        rows=torch.as_tensor(saved['rows'][start:start+16],device=device)
        query=torch.as_tensor(saved['queries'][start:start+16],device=device)
        history=torch.as_tensor(saved['history'][start:start+16],device=device)
        items=torch.as_tensor(saved['candidates'][start:start+16],device=device)
        b=len(rows)
        z=model(query[:,None,:].expand(b,k,-1).reshape(b*k,-1),rows[:,None].expand(b,k).reshape(-1),
                items.reshape(-1),history[:,None,:].expand(b,k,-1).reshape(b*k,-1))
        scores.append(z.reshape(b,k).cpu().numpy())
    scores=np.concatenate(scores)
    order=np.argsort(-scores,axis=1,kind='stable')
    ranked=np.take_along_axis(saved['candidates'],order,axis=1)
    data=PreparedData(config['cache'])
    metrics,per=reranking_metrics(ranked,saved['candidates'],saved['targets'],saved['rows'],data.train_counts)
    arrays=dict(predictions=ranked,ranking_scores=np.take_along_axis(scores,order,axis=1),
                rows=saved['rows'],targets=saved['targets'],**per)
    return dict(ranking=metrics,observed=classification),arrays,dict(
        logits=logits,labels=point['labels'].cpu().numpy(),rows=point['rows'].cpu().numpy())


def load_ranker(checkpoint,device):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);config=saved['configuration']
    folder=Path(config['root'])/f'features_seed{saved["seed"]}'
    assert digest(folder/'metadata.json')==saved['features_metadata_sha256']
    model=model_for(config,saved['seed'],saved['variant']);model.load_state_dict(saved['model'])
    return model.to(device).eval(),config,saved


def train(args):
    config=read(args.config);t=config['training'];root=Path(config['root'])
    out=root/'training'/f'{args.variant}_seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    if (out/'config.json').exists():raise FileExistsError('Incomplete run; choose a new root.')
    seed_everything(args.seed);model=model_for(config,args.seed,args.variant)
    initial=initialization_hash(model)
    info=dict(configuration=config,seed=args.seed,variant=args.variant,initialization_sha256=initial,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        features_metadata_sha256=digest(root/f'features_seed{args.seed}/metadata.json'),test_evaluated=False)
    write_new(out/'config.json',info);model=model.to(args.device)
    train=tensors(config,args.seed,'train',args.device);point=tensors(config,args.seed,'pointval',args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
    rng=torch.Generator(device=args.device).manual_seed(args.seed+111)
    best=(-1.,-1.);history=[]
    for epoch in range(1,t['epochs']+1):
        model.train();start=time.perf_counter();loss_sum=0.
        order=torch.randperm(len(train['rows']),device=args.device,generator=rng)
        for indices in order.split(t['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            logits=model(train['queries'][indices],train['rows'][indices],train['items'][indices],train['history'][indices])
            loss=F.binary_cross_entropy_with_logits(logits,train['labels'][indices].float())
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite ranker loss.')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),t['grad_clip'],error_if_nonfinite=True)
            optimizer.step();loss_sum+=float(loss.detach())*len(indices)
        seconds=time.perf_counter()-start
        metrics,arrays,events=evaluate(model,config,args.seed,args.device,point)
        np.savez_compressed(out/f'validation_epoch{epoch}.npz',**arrays)
        np.savez_compressed(out/f'observed_epoch{epoch}.npz',**events)
        row=dict(epoch=epoch,training_loss=loss_sum/len(order),training_seconds=seconds,validation=metrics)
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps(dict(event='epoch',seed=args.seed,variant=args.variant,**row)),flush=True)
        score=metrics['ranking']['NDCG10'],metrics['ranking']['HR10']
        if score>best:
            best=score
            torch.save(dict(model=model.state_dict(),epoch=epoch,configuration=config,seed=args.seed,variant=args.variant,
                validation=metrics,features_metadata_sha256=info['features_metadata_sha256']),out/'best.pt')
    chosen=max(history,key=lambda h:(h['validation']['ranking']['NDCG10'],h['validation']['ranking']['HR10'],-h['epoch']))
    write_new(out/'result.json',dict(seed=args.seed,variant=args.variant,best_epoch=chosen['epoch'],
        validation=chosen['validation'],fixed_epoch5=history[-1]['validation'],
        initialization_sha256=initial,parameters=info['trainable_parameters'],
        training_seconds=sum(h['training_seconds'] for h in history),
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated(args.device)/1024**2,
        checkpoint_sha256=digest(out/'best.pt'),test_evaluated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/ranking_study.json')
    p.add_argument('--variant',choices=['mean_mlp','din_attention'],required=True)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--device',default='cuda:5')
    train(p.parse_args())
