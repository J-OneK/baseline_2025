"""Independent content feature learning and fixed score blending."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from taac_retrieval.content_ranker import ContentRanker,blend_scores


class ContentRankerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(5)
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name);cache=root/'data';cache.mkdir()
        meta=dict(item_fields=['100','121'],user_fields=['103'],user_array_fields=['106'],
            feature_sizes={'100':5,'121':400001,'103':4,'106':5})
        (cache/'metadata.json').write_text(json.dumps(meta))
        values=dict(item_sparse=np.array([[0,0],[1,10],[2,20],[3,30],[1,40]]),
            item_mm81=np.random.default_rng(1).normal(size=(5,32)).astype('float32'),
            mm_present=np.array([False,True,True,True,False]),user_sparse=np.array([[0],[1],[2]]),
            user_array_106=np.array([[0,0],[1,0],[2,3]]))
        for k,v in values.items():np.save(cache/f'{k}.npy',v)
        ex=root/'ref/examples';ex.mkdir(parents=True)
        np.save(ex/'stats_events.npy',np.array([0,10,5,2,1]));np.save(ex/'stats_clicks.npy',np.array([0,1,1,0,0]))
        (ex/'metadata.json').write_text(json.dumps(dict(statistics_global_ctr=.1,summary=dict(train=dict(click_rate=.1)))))
        base=root/'base.json';base.write_text(json.dumps(dict(cache=str(cache))))
        self.config=dict(base=str(base),reference_root=str(root/'ref'),
            model=dict(feature_embedding=4,hidden=4,max_item_field_cardinality=10,dropout=0.))

    def tearDown(self):self.temp.cleanup()

    def test_retrieval_queries_do_not_enter_classifier(self):
        model=ContentRanker(self.config).eval()
        with torch.no_grad():model.prediction[-1].weight.normal_(std=.1)
        row=torch.tensor([0,1]);item=torch.tensor([1,2]);history=torch.tensor([[2,3,0],[1,4,0]])
        a=model(torch.randn(2,4),row,item,history)
        b=model(torch.randn(2,4)*100,row,item,history)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertEqual(model.embedded_fields,['100']);self.assertNotIn('121',model.embedding)

    def test_content_and_attention_learn_without_retrieval_vectors(self):
        model=ContentRanker(self.config)
        with torch.no_grad():model.prediction[-1].weight.normal_(std=.1)
        q=torch.randn(2,4);rows=torch.tensor([0,1]);items=torch.tensor([1,2]);hist=torch.tensor([[2,3,0],[1,4,0]])
        z,weights=model(q,rows,items,hist,True)
        torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor([1.,0.])).backward()
        self.assertGreater(float(model.embedding['100'].weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.mm_projection.weight.grad.abs().sum()),0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.attention.parameters()),0.)
        self.assertEqual(weights[:,2].tolist(),[0.,0.])
        self.assertEqual(float(model.embedding['100'].weight.grad[0].abs().sum()),0.)

    def test_modes_share_full_initialization_and_empty_history_is_finite(self):
        torch.manual_seed(8);a=ContentRanker(self.config,'mean_mlp')
        torch.manual_seed(8);b=ContentRanker(self.config,'din_attention')
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y)
        b.eval();z=b(torch.zeros(2,4),torch.tensor([0,1]),torch.tensor([1,2]),torch.zeros(2,3,dtype=torch.long))
        self.assertTrue(torch.isfinite(z).all())

    def test_blending_is_finite_for_constant_scores_and_preserves_endpoints(self):
        q=np.array([[1.,3.,2.],[4.,4.,4.]])
        p=np.array([[2.,1.,3.],[0.,0.,0.]])
        self.assertTrue(np.isfinite(blend_scores(q,p,.1)).all())
        self.assertEqual(np.argsort(blend_scores(q,p,0)[0]).tolist(),np.argsort(q[0]).tolist())
        self.assertEqual(np.argsort(blend_scores(q,p,1)[0]).tolist(),np.argsort(p[0]).tolist())
        np.testing.assert_allclose(blend_scores(q,p,.1)[:1],blend_scores(q[:1],p[:1],.1))
        with self.assertRaises(ValueError):blend_scores(q,p,2)


if __name__=='__main__':unittest.main()
