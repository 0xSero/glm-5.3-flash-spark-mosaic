import json,unittest
from pathlib import Path
from build_plan import allocation,N
class AllocationTests(unittest.TestCase):
 def test_budget_and_determinism(self):
  scores={3:list(range(1,N+1)),4:list(range(1,N+1))};rank={l:list(range(N-1,-1,-1)) for l in scores}
  a=allocation(scores,rank,budget=16,floor=272);self.assertEqual(a,allocation(scores,rank,budget=16,floor=272));self.assertEqual(a[0],{3:280,4:280})
 def test_per_layer_scale_invariance(self):
  scores={3:list(range(1,N+1)),4:list(range(1,N+1))};rank={l:list(range(N-1,-1,-1)) for l in scores}
  x=allocation(scores,rank,budget=32,floor=256)[0];scores[3]=[v*8 for v in scores[3]]
  self.assertEqual(x,allocation(scores,rank,budget=32,floor=256)[0])
 def test_invalid_inputs(self):
  s={3:[1]*N};r={3:list(range(N))}
  for kw in ({'budget':1},{'budget':64,'floor':272},{'floor':239}):
   with self.assertRaises(ValueError):allocation(s,r,**kw)
  with self.assertRaises(ValueError):allocation({3:[-1]*N},r,budget=8)
 def test_sealed_plans_match_memory_and_map(self):
  root=Path(__file__).with_name('plan-v1');plans=[json.loads((root/(n+'.json')).read_text()) for n in ('uniform272','nonuniform272budget')]
  for p in plans:
   c=p['text_config_patch'];self.assertEqual(c['n_routed_experts'],288);self.assertEqual(set(c['routed_experts_per_layer']),set(map(str,range(3,45))))
   self.assertEqual(set(c['routed_experts_per_layer']),set(c['retained_expert_ids_by_layer']))
   self.assertEqual(sum(c['routed_experts_per_layer'].values()),11424)
   for l,n in c['routed_experts_per_layer'].items():
    ids=c['retained_expert_ids_by_layer'][l];self.assertEqual(ids,sorted(set(ids)));self.assertEqual(len(ids),n);self.assertEqual(n%8,0);self.assertGreaterEqual(n,240)
    self.assertEqual(p['layer_details'][l]['original_id_to_contiguous_id'],{str(e):i for i,e in enumerate(ids)})
   self.assertEqual(p['native_mtp_n_routed_experts'],288);self.assertFalse(p['selection_uses_heldout_quality'])
  self.assertEqual(plans[0]['projected_tensor_bytes'],plans[1]['projected_tensor_bytes'])
  self.assertEqual(plans[0]['retained_router_bytes'],plans[1]['retained_router_bytes'])
if __name__=='__main__':unittest.main()
