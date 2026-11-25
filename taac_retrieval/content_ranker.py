"""Independent content ranker; DIN-inspired attention, no retrieval embeddings."""
import json,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .ranking_model import masked_attention
from .ranking_data import read


class ContentRanker(nn.Module):
    def __init__(self,config,variant='din_attention'):
        super().__init__()
        base=read(config['base']);meta=read(Path(base['cache'])/'metadata.json')
        self.variant=variant;self.item_fields=meta['item_fields'];self.user_fields=meta['user_fields'];self.array_fields=meta['user_array_fields']
        m=config['model'];e=m['feature_embedding'];h=m['hidden']
        self.embedded_fields=[f for f in self.item_fields if meta['feature_sizes'][f]<=m['max_item_field_cardinality']]
        for name in ['item_sparse','item_mm81','mm_present','user_sparse']+[f'user_array_{f}' for f in self.array_fields]:
            self.register_buffer(name,torch.as_tensor(np.load(Path(base['cache'])/f'{name}.npy')),persistent=False)
        ex=Path(config['reference_root'])/'examples'
        for name,file in [('event_counts','stats_events'),('click_counts','stats_clicks')]:
            self.register_buffer(name,torch.as_tensor(np.load(ex/f'{file}.npy')),persistent=False)
        em=read(ex/'metadata.json');self.register_buffer('global_ctr',torch.tensor(em['statistics_global_ctr']))
        fields=self.embedded_fields+self.user_fields+self.array_fields
        self.embedding=nn.ModuleDict({f:nn.Embedding(meta['feature_sizes'][f],e,padding_idx=0) for f in fields})
        for embedding in self.embedding.values():
            nn.init.normal_(embedding.weight,std=.02)
            with torch.no_grad():embedding.weight[0].zero_()
        self.mm_projection=nn.Linear(32,h)
        self.item_projection=nn.Sequential(nn.Linear(len(self.embedded_fields)*e+h,h),nn.PReLU(),nn.LayerNorm(h))
        self.user_projection=nn.Sequential(nn.Linear((len(self.user_fields)+len(self.array_fields))*e,h),nn.PReLU(),nn.LayerNorm(h))
        self.attention=nn.Sequential(nn.Linear(h*4,64),nn.PReLU(),nn.Linear(64,32),nn.PReLU(),nn.Linear(32,1))
        dim=h*5+len(self.item_fields)+5
        self.prediction=nn.Sequential(nn.Linear(dim,128),nn.PReLU(),nn.Dropout(m['dropout']),
            nn.Linear(128,64),nn.PReLU(),nn.Dropout(m['dropout']),nn.Linear(64,1))
        nn.init.zeros_(self.prediction[-1].weight)
        p=em['summary']['train']['click_rate'];nn.init.constant_(self.prediction[-1].bias,math.log(p/(1-p)))

    def encode_items(self,ids):
        attr=self.item_sparse[ids.long()]
        parts=[self.embedding[f](attr[:,self.item_fields.index(f)].long()) for f in self.embedded_fields]
        parts.append(self.mm_projection(self.item_mm81[ids.long()]))
        return self.item_projection(torch.cat(parts,-1))*(ids>0)[:,None]

    def encode_users(self,rows):
        ids=rows.long()+1;attr=self.user_sparse[ids]
        parts=[self.embedding[f](attr[:,j].long()) for j,f in enumerate(self.user_fields)]
        for f in self.array_fields:
            array=getattr(self,f'user_array_{f}')[ids].long()
            parts.append(self.embedding[f](array).sum(1))
        return self.user_projection(torch.cat(parts,-1))

    def forward(self,queries,rows,items,history,return_attention=False):
        # queries deliberately unused: no retrieval training representations in CTR learning.
        valid=history!=0
        all_ids=torch.cat((items.reshape(-1),history.reshape(-1))).long()
        unique,inverse=torch.unique(all_ids,return_inverse=True)
        encoded=self.encode_items(unique);n=len(items)
        v=encoded[inverse[:n]];h=encoded[inverse[n:]].reshape(len(items),history.shape[1],-1)
        u=self.encode_users(rows)
        if self.variant=='din_attention':
            target=v[:,None,:].expand_as(h)
            weight=masked_attention(self.attention(torch.cat((h,target,h*target,h-target),-1)).squeeze(-1),valid)
        else:weight=valid.float()/valid.sum(-1,keepdim=True).clamp_min(1)
        interest=(h*weight[:,:,None]).sum(1)
        attr=self.item_sparse[items.long()];histattr=self.item_sparse[history.long()]
        available=valid[:,:,None]&(attr[:,None,:]>0)&(histattr>0)
        matches=((attr[:,None,:]==histattr)&available).sum(1)/available.sum(1).clamp_min(1)
        events=self.event_counts[items.long()].float();clicks=self.click_counts[items.long()].float()
        prior=(clicks+20*self.global_ctr)/(events+20)
        scalar=torch.stack((events.log1p()/10,clicks.log1p()/10,
            torch.logit(prior.clamp(1e-6,1-1e-6))-torch.logit(self.global_ctr),
            valid.float().mean(-1),self.mm_present[items.long()].float()),-1)
        output=self.prediction(torch.cat((u,v,interest,u*v,(interest-v).abs(),matches,scalar),-1)).squeeze(-1)
        return (output,weight) if return_attention else output


def standardize(scores):
    mean=scores.mean(1,keepdims=True);std=scores.std(1,keepdims=True)
    return (scores-mean)/np.where(std<1e-6,1.,std)


def blend_scores(recall,ctr,alpha=.1):
    if not 0<=alpha<=1:raise ValueError('alpha outside [0,1]')
    return (1-alpha)*standardize(recall)+alpha*standardize(ctr)
