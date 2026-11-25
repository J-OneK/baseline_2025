"""Ranker protocol: causal events, attention, residual initialization and candidate integrity."""
import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from taac_retrieval.data import make_input
from taac_retrieval.ranking_data import eligible_events,history_clicks,gather_queries
from taac_retrieval.ranking_model import FrozenFeatures,RankingHead,masked_attention
from taac_retrieval.ranking_metrics import observed_metrics,reranking_metrics


def features():
    torch.manual_seed(9)
    items=torch.randn(510,4);items[0]=0
    users=torch.randn(4,4);users[0]=0
    attr=torch.randint(1,5,(510,3));attr[0]=0
    events=torch.arange(510);clicks=events//4
    return FrozenFeatures(items,users,attr,events,clicks,0.,1.,.1)


class CumulativeEncoder(nn.Module):
    hidden=2
    def encode_history(self,items,kinds,actions,user):
        value=items.float()*(kinds==1)+user[:,None]*10*(kinds==2)
        return value.cumsum(1)[:,:,None].expand(-1,-1,2)


class RankingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_logged_labels_unknowns_repeats_and_statistic_period(self):
        items=np.array([1,2,3,4,5,6,6,7,8,9])
        actions=np.array([1,2,3,1,2,1,2,3,2,2])
        times=np.array([1,2,3,4,5,6,7,8,9,10])
        cut,stats,train,val=eligible_events(items,actions,times,8,9,8,dict(min_history=3))
        self.assertEqual(cut,4);self.assertEqual(stats,[0,1,3])
        self.assertEqual(train,[(4,4),(5,5)])
        self.assertEqual(val,[(8,8)])  # test position9 absent; unknown7 absent
        self.assertFalse(set(stats)&set(p for p,c in train))

    def test_timestamp_ties_cannot_enter_history(self):
        x=np.arange(1,11);times=np.array([1,2,3,4,5,6,6,7,8,9])
        _,_,train,_=eligible_events(x,np.ones(10,int),times,8,9,8,dict(min_history=3))
        self.assertEqual(dict(train)[6],5)
        self.assertEqual(history_clicks(x,np.full(10,2),5,3).tolist(),[3,4,5])

    def test_cached_states_equal_truncated_queries_and_ignore_future(self):
        data=SimpleNamespace(num_users=1,max_history=6,train_cut=np.array([4]),test_cut=np.array([6]),
            items=np.array([[1,2,3,4,5,6]]),actions=np.array([[1,2,1,2,1,2]]))
        encoder=CumulativeEncoder()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);np.save(root/'train_rows.npy',np.array([0,0]));np.save(root/'train_cuts.npy',np.array([1,3]))
            a=gather_queries(encoder,data,root,'train','cpu')
            for i,cut in enumerate([1,3]):
                seq,kinds,acts=make_input(data.items[0,:cut],data.actions[0,:cut],6)
                expected=encoder.encode_history(torch.tensor(seq)[None],torch.tensor(kinds)[None],
                    torch.tensor(acts)[None],torch.tensor([1]))[:,-1].numpy()[0]
                np.testing.assert_array_equal(a[i],expected)
            data.items[0,3:]=99
            np.testing.assert_array_equal(a,gather_queries(encoder,data,root,'train','cpu'))

    def test_empty_history_attention_and_gradients_are_finite(self):
        logits=torch.randn(2,3,requires_grad=True);valid=torch.tensor([[False]*3,[True,False,True]])
        weight=masked_attention(logits,valid)
        self.assertEqual(weight[0].sum().item(),0.)
        self.assertEqual(weight[1].sum().item(),1.)
        weight.square().sum().backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_initial_ranker_is_monotonic_recall_and_modes_share_weights(self):
        torch.manual_seed(1);a=RankingHead(features(),'mean_mlp',hidden=4,dropout=0.).eval()
        torch.manual_seed(1);b=RankingHead(features(),'din_attention',hidden=4,dropout=0.).eval()
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y)
        q=torch.randn(3,4);rows=torch.zeros(3,dtype=torch.long);items=torch.tensor([1,2,3]);hist=torch.zeros(3,2,dtype=torch.long)
        raw=(q*a.features.items[items]).sum(-1)
        output=a(q,rows,items,hist);other=b(q,rows,items,hist)
        torch.testing.assert_close(output,other)
        self.assertEqual(output.argsort().tolist(),raw.argsort().tolist())

    def test_attention_changes_with_target_and_never_reads_padding(self):
        model=RankingHead(features(),'din_attention',hidden=4,dropout=0.).eval()
        q=torch.ones(2,4);rows=torch.zeros(2,dtype=torch.long);items=torch.tensor([1,5]);hist=torch.tensor([[2,3,0],[2,3,0]])
        _,weights=model(q,rows,items,hist,True)
        self.assertEqual(weights[:,2].tolist(),[0.,0.])
        self.assertGreater(float((weights[0]-weights[1]).abs().max().detach()),1e-7)

    def test_attention_receives_gradients_after_residual_head_opens(self):
        model=RankingHead(features(),'din_attention',hidden=4,dropout=0.)
        with torch.no_grad():model.prediction[-1].weight.fill_(.05)
        q=torch.randn(3,4);rows=torch.zeros(3,dtype=torch.long);items=torch.tensor([1,4,5]);hist=torch.tensor([[2,3],[2,4],[1,3]])
        model(q,rows,items,hist).square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.attention.parameters()),0.)

    def test_candidates_cannot_inject_target_and_unrecalled_users_stay(self):
        candidates=np.tile(np.arange(1,501),(2,1))
        target=np.array([499,501]);rows=np.array([1,2]);counts=np.ones(502,int)
        ranked=candidates[:,::-1].copy()
        m,p=reranking_metrics(ranked,candidates,target,rows,counts)
        self.assertEqual(m['HR10'],.5);self.assertEqual(m['Recall500'],.5);self.assertEqual(m['conditional_HR10'],1.)
        ranked[1,0]=501
        with self.assertRaises(ValueError):reranking_metrics(ranked,candidates,target,rows,counts)

    def test_observed_metrics_stable_logits_and_not_candidate_labels(self):
        m=observed_metrics(np.array([0,1]),np.array([-1000.,1000.]),np.array([1,2]))
        self.assertEqual(m['AUC'],1.);self.assertEqual(m['LogLoss'],0.)


if __name__=='__main__':
    unittest.main()
