#!/usr/bin/env python3
"""CPU-only predeclared equal-payload REAP allocation from sealed observations."""
import argparse,csv,hashlib,heapq,json,math
from pathlib import Path
LAYERS=tuple(range(3,45));N=288;UNIFORM=272;FLOOR=240;BLOCK=8;REMOVE=(N-UNIFORM)*len(LAYERS)
EXPECTED={'seal.json':'bb2d12f209d2606319a0f1e3c40b0f54db2d4d3e9d40cfe49779b510f3c5596a','observations.json':'b2e36095306132b24a68accd0263fa42cf319d335b1533e4c85e9348a618cabf','candidates.json':'4fd8de1e8d0d3e05158b0989a6b18813430f30b280f5499870e7e75b06d02ecd'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,message):
 if not ok:raise ValueError(message)
def allocation(scores,rankings,budget=REMOVE,floor=FLOOR,block=BLOCK):
 require(budget%block==0,'budget must be divisible by block')
 require(8<=floor<=N and floor%block==0,'invalid floor')
 require(0<=budget<=len(scores)*(N-floor),'infeasible removal budget')
 counts={layer:N for layer in scores};normalized={};heap=[];ledger=[]
 for layer in sorted(scores):
  values=scores[layer];rank=rankings[layer]
  require(len(values)==N and sorted(rank)==list(range(N)),'invalid expert coverage')
  require(all(math.isfinite(v) and v>=0 for v in values) and sum(values)>0,'invalid scores')
  require(rank==sorted(range(N),key=lambda e:(-values[e],e)),'ranking differs from massmax criterion')
  normalized[layer]=[v/sum(values) for v in values]
  if N>floor:heapq.heappush(heap,(sum(normalized[layer][e] for e in rank[N-block:N]),layer))
 for step in range(budget//block):
  cost,layer=heapq.heappop(heap);old=counts[layer];new=old-block;counts[layer]=new
  ledger.append({'step':step,'layer':layer,'from':old,'to':new,'removed_original_ids':sorted(rankings[layer][new:old]),'normalized_proxy_removed':cost})
  if new>floor:heapq.heappush(heap,(sum(normalized[layer][e] for e in rankings[layer][new-block:new]),layer))
 return counts,ledger,normalized

def build(args):
 root=args.root;sealed=root/'sealed-observations';inputs={}
 for n,digest in EXPECTED.items():
  p=sealed/n;require(sha(p)==digest,'sealed observations changed');inputs['sealed-observations/'+n]=digest
 seal=read(sealed/'seal.json');obs=read(sealed/'observations.json');cand=read(sealed/'candidates.json')
 require(seal['artifact_sha256']=={k:v for k,v in EXPECTED.items() if k!='seal.json'},'seal dependency mismatch')
 require(seal['minimum_expert_routes']>0 and all(not x for x in obs['unseen_experts'].values()),'incomplete natural route coverage')
 rows=cand['selection']['massmax_domain'];require(set(rows)==set(map(str,LAYERS)),'wrong layer coverage')
 scores={};rankings={};domain_scores={}
 for layer in LAYERS:
  key=str(layer);scores[layer]=rows[key]['scores'];rankings[layer]=rows[key]['ranked_experts_high_to_low'];domain_scores[layer]={}
  for domain,group in obs['domain_observations'].items():
   mass=group[key]['weighted_ean_sum'];denom=sum(mass);require(denom>0,'empty domain mass')
   domain_scores[layer][domain]=[v/denom for v in mass]
  recomputed=[max(s[e] for s in domain_scores[layer].values()) for e in range(N)]
  require(all(math.isclose(a,b,rel_tol=1e-13,abs_tol=1e-15) for a,b in zip(scores[layer],recomputed)),'sealed score does not reproduce')
 nonuniform,ledger,normalized=allocation(scores,rankings);uniform={l:UNIFORM for l in LAYERS}
 audit_path=root/'quality/mixed-precision-feasibility/header-byte-audit.json';headers=read(audit_path);expert_bytes=headers['2']['tensor_bytes'];require(expert_bytes==6402096,'unexpected K2 expert payload')
 router_path=Path(__file__).with_name('router-header-audit.json');router=read(router_path);router_rows={}
 for l in LAYERS:
  weight=router[f'model.language_model.layers.{l}.mlp.gate.weight'];bias=router[f'model.language_model.layers.{l}.mlp.gate.e_score_correction_bias']
  require(weight['shape']==[288,4096] and weight['dtype']=='BF16' and bias['shape']==[288] and bias['dtype']=='F32','router geometry changed')
  router_rows[l]=(weight['tensor_bytes']+bias['tensor_bytes'])//N
 require(set(router_rows.values())=={8196},'router row costs differ')
 k2_path=root/'source-inventory-k2/EXL3_MANIFEST.json';k2=read(k2_path)
 require(sha(k2_path)=='501641b947fa56afc9ed098bdc034e59c2bd229d4fa1a1d7b80e3727f10c8ba3','wrong K2 source')
 require(k2['physical_tensor_bytes']==111352026456 and k2['retained_tensor_bytes']==33835039608,'source file/retained payload changed')
 source_tensor_bytes=expert_bytes*N*len(LAYERS)+k2['retained_tensor_bytes'];require(source_tensor_bytes==111274792824,'derived source tensor payload changed')
 for p in (audit_path,router_path,k2_path):inputs[str(p.relative_to(root))]=sha(p)
 out=args.output;require(not out.exists(),'refuse overwrite');out.mkdir(parents=True)
 plans={};summary=[]
 for label,counts in [('uniform272',uniform),('nonuniform272budget',nonuniform)]:
  ids={str(l):sorted(rankings[l][:counts[l]]) for l in LAYERS};pruned={str(l):sorted(rankings[l][counts[l]:]) for l in LAYERS}
  removed=sum(N-n for n in counts.values());require(removed==REMOVE,'unequal expert budget')
  reduction=sum((N-counts[l])*(expert_bytes+router_rows[l]) for l in LAYERS)
  metadata={'n_routed_experts':288,'routed_experts_per_layer':{str(l):counts[l] for l in LAYERS},'retained_expert_ids_by_layer':ids}
  detail={}
  for l in LAYERS:
   kept=ids[str(l)];pool=obs['observations'][str(l)]['weighted_ean_sum'];min_ret=min(sum(s[e] for e in kept) for s in domain_scores[l].values())
   detail[str(l)]={'keep':counts[l],'normalized_proxy_removed':sum(normalized[l][e] for e in pruned[str(l)]),'pooled_mass_retention':sum(pool[e] for e in kept)/sum(pool),'minimum_domain_mass_retention':min_ret,'original_id_to_contiguous_id':{str(e):i for i,e in enumerate(kept)}}
  plan={'schema':'glm53-nonuniform-pruning-plan-v1','state':'PREDECLARED_CPU_PLAN_NOT_BUILT','name':label,'text_config_patch':metadata,'native_mtp_n_routed_experts':288,'pruned_original_ids_by_layer':pruned,'layer_details':detail,'source_k2_repo':'0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw','source_k2_revision':'35b4b580debe2a1510a2eb042d9d6d4cd7fb3a6b','source_k2_manifest_sha256':sha(k2_path),'source_observation_model_identity':cand['model_identity'],'ranking':'massmax_domain','allocation_proxy':'Within-layer massmax_domain normalized to unit sum; cross-layer sum is heuristic, not measured counterfactual sensitivity.','selection_uses_heldout_quality':False,'expert_payload_bytes_each':expert_bytes,'router_row_bytes_each':8196,'removed_experts':removed,'retained_experts':sum(counts.values()),'removed_tensor_bytes':reduction,'projected_tensor_bytes':source_tensor_bytes-reduction,'unchanged_native_nonrouter_bytes':k2['retained_tensor_bytes']-sum(router_rows.values())*N,'retained_router_bytes':sum(counts[l]*router_rows[l] for l in LAYERS),'protected_policy':'Retain every other native tensor byte-exact, including all889 MTP tensors/all288 MTP experts; only target router rows follow retained original IDs.','quality_accepted':False,'runtime_accepted':False,'inputs_sha256':inputs}
  plans[label]=plan;(out/(label+'.json')).write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
 require(plans['uniform272']['projected_tensor_bytes']==plans['nonuniform272budget']['projected_tensor_bytes'],'unequal payload')
 for l in LAYERS:summary.append({'layer':l,'uniform_keep':UNIFORM,'nonuniform_keep':nonuniform[l],'massmax_score_sum':sum(scores[l]),'uniform_removed_proxy':plans['uniform272']['layer_details'][str(l)]['normalized_proxy_removed'],'nonuniform_removed_proxy':plans['nonuniform272budget']['layer_details'][str(l)]['normalized_proxy_removed']})
 with (out/'layer-summary.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
 (out/'allocation-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
 validation={'state':'CPU_PLAN_VALIDATED_NOT_BUILT','expert_counts_equal':True,'tensor_bytes_equal_including_router':True,'projected_tensor_bytes':plans['uniform272']['projected_tensor_bytes'],'retained_experts':sum(nonuniform.values()),'removed_experts':REMOVE,'minimum_nonuniform_count':min(nonuniform.values()),'maximum_nonuniform_count':max(nonuniform.values()),'normalized_proxy_removed_uniform':sum(x['uniform_removed_proxy'] for x in summary),'normalized_proxy_removed_nonuniform':sum(x['nonuniform_removed_proxy'] for x in summary),'raw_massmax_score_sum_range':[min(map(sum,scores.values())),max(map(sum,scores.values()))],'minimum_natural_expert_routes':seal['minimum_expert_routes'],'records':seal['records'],'tokens':seal['tokens'],'missing_domains':seal['missing_domains'],'observations_state':seal['state'],'observation_domains':sorted(obs['domain_observations']),'builder_sha256':sha(__file__),'output_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}}
 (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);build(p.parse_args())
