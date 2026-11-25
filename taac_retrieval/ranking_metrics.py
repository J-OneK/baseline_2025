"""Separate observed-label classification from fixed-candidate ranking metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score
from .coverage import coverage_summary


def observed_metrics(labels,logits,rows):
    labels=np.asarray(labels);logits=np.asarray(logits,dtype=np.float64)
    if labels.shape!=logits.shape or not np.isin(labels,[0,1]).all() or not np.isfinite(logits).all():
        raise ValueError('Expected finite logits and observed binary labels.')
    return dict(events=len(labels),users=len(np.unique(rows)),clicks=int(labels.sum()),
        observed_click_rate=float(labels.mean()),
        AUC=float(roc_auc_score(labels,logits)) if len(np.unique(labels))==2 else None,
        LogLoss=float((np.logaddexp(0,logits)-labels*logits).mean()),
        scope='Selected observed first-appearance events after train_cut and strictly before test_cut; not retrieved-unexposed negatives')


def reranking_metrics(predictions,candidates,targets,rows,counts):
    predictions,candidates=map(np.asarray,(predictions,candidates))
    if predictions.shape!=candidates.shape or not np.array_equal(np.sort(predictions,1),np.sort(candidates,1)):
        raise ValueError('Ranking must only permute original candidates; target injection is forbidden.')
    groups,per=coverage_summary(predictions,targets,rows,counts,ks=(10,100,500))
    eligible=per['hits_at_500'].astype(bool)
    result=dict(HR10=groups['all']['Recall@10'],NDCG10=groups['all']['NDCG@10'],
        Recall500=groups['all']['Recall@500'],users=len(rows),retrieval_k=candidates.shape[1],
        covered_users=int(eligible.sum()),conditional_HR10=float(per['hits_at_10'][eligible].mean()) if eligible.any() else None,
        conditional_NDCG10=float(per['ndcg_at_10'][eligible].mean()) if eligible.any() else None,groups=groups)
    return result,per
