"""Small residual ranking heads over frozen retrieval features."""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


class FrozenFeatures(nn.Module):
    def __init__(self,items,users,attributes,events,clicks,mean,std,global_ctr):
        super().__init__()
        for name,value in [('items',items),('users',users),('attributes',attributes),
                           ('events',events),('clicks',clicks)]:
            self.register_buffer(name,torch.as_tensor(value),persistent=False)
        self.register_buffer('score_mean',torch.tensor(float(mean)))
        self.register_buffer('score_std',torch.tensor(max(float(std),1e-6)))
        self.register_buffer('global_ctr',torch.tensor(float(global_ctr)))

    @classmethod
    def load(cls,config,seed):
        root=Path(config['root']);folder=root/f'features_seed{seed}';examples=root/'examples'
        meta=json.loads((folder/'metadata.json').read_text())
        summary=json.loads((examples/'metadata.json').read_text())
        return cls(np.load(folder/'item_vectors.npy'),np.load(folder/'user_vectors.npy'),
            np.load(Path(config['cache'])/'item_sparse.npy'),np.load(examples/'stats_events.npy'),
            np.load(examples/'stats_clicks.npy'),meta['raw_score_normalization']['mean'],
            meta['raw_score_normalization']['std'],summary['statistics_global_ctr'])


def masked_attention(logits,valid):
    weights=logits.masked_fill(~valid,-1e9).softmax(-1)*valid
    return weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)


class RankingHead(nn.Module):
    def __init__(self,features,variant='din_attention',hidden=32,dropout=.1,initial_ctr=.1):
        super().__init__()
        if variant not in ['mean_mlp','din_attention']:raise ValueError(variant)
        self.features,self.variant=features,variant
        self.query_norm=nn.LayerNorm(hidden);self.item_norm=nn.LayerNorm(hidden);self.user_norm=nn.LayerNorm(hidden)
        # Allocate the identical network for both arms; only pooling differs.
        self.attention=nn.Sequential(nn.Linear(hidden*4,64),nn.PReLU(),nn.Linear(64,32),nn.PReLU(),nn.Linear(32,1))
        dimension=hidden*6+features.attributes.shape[1]+5
        self.prediction=nn.Sequential(nn.Linear(dimension,128),nn.PReLU(),nn.Dropout(dropout),
            nn.Linear(128,64),nn.PReLU(),nn.Dropout(dropout),nn.Linear(64,1))
        nn.init.zeros_(self.prediction[-1].weight);nn.init.zeros_(self.prediction[-1].bias)
        self.recall_weight=nn.Parameter(torch.tensor(math.log(math.expm1(1.))))
        self.intercept=nn.Parameter(torch.tensor(math.log(initial_ctr/(1-initial_ctr))))

    def forward(self,queries,rows,items,history,return_attention=False):
        f=self.features;valid=history!=0
        item=f.items[items.long()];hist=f.items[history.long()]
        q=self.query_norm(queries);v=self.item_norm(item);h=self.item_norm(hist)*valid[:,:,None]
        u=self.user_norm(f.users[rows.long()+1])
        if self.variant=='din_attention':
            target=v[:,None,:].expand_as(h)
            local=torch.cat((h,target,h*target,h-target),-1)
            weight=masked_attention(self.attention(local).squeeze(-1),valid)
        else:
            weight=valid.float()/valid.sum(-1,keepdim=True).clamp_min(1)
        interest=(h*weight[:,:,None]).sum(1)
        attr=f.attributes[items.long()]
        hist_attr=f.attributes[history.long()]
        available=valid[:,:,None]&(attr[:,None,:]>0)&(hist_attr>0)
        match=((attr[:,None,:]==hist_attr)&available).sum(1)/available.sum(1).clamp_min(1)
        events=f.events[items.long()].float();clicks=f.clicks[items.long()].float()
        prior=(clicks+20*f.global_ctr)/(events+20)
        prior_logit=torch.logit(prior.clamp(1e-6,1-1e-6))-torch.logit(f.global_ctr)
        raw=(queries*item).sum(-1)
        numeric=torch.stack((events.log1p()/10,clicks.log1p()/10,prior_logit,
                             raw/10,valid.float().mean(-1)),-1)
        x=torch.cat((q,u,v,q*v,(q-v).abs(),interest,match,numeric),-1)
        correction=self.prediction(x).squeeze(-1)
        output=self.intercept+F.softplus(self.recall_weight)*(raw-f.score_mean)/f.score_std+correction
        return (output,weight) if return_attention else output
