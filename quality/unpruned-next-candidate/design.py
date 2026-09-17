#!/usr/bin/env python3
"""Create unpruned allocation designs from independent calibration only."""
import hashlib,heapq,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def expert_plan(rows,budget=576):
 counts={l:0 for l in rows};heap=[]
 for l,experts in rows.items():heapq.heappush(heap,(-sum(x['measured_error_reduction'] for x in experts[:8]),l))
 for _ in range(budget//8):
  _,l=heapq.heappop(heap);counts[l]+=8;n=counts[l]
  if n<288:heapq.heappush(heap,(-sum(x['measured_error_reduction'] for x in rows[l][n:n+8]),l))
 return {str(l):sorted(x['expert'] for x in rows[l][:n]) for l,n in sorted(counts.items())}
def main():
 invp=ROOT/'quality/mixed-layer-candidate/paired-proxy-inventory.json';inv=load(invp)
 assert sha(invp)=='78e602f9ee18d25a4a5334d0ba09564f2a90d6601e81fb4e65be9972e5b16dd3'
 rows={};sources={};sidecar_hashes={}
 for ref in inv['sidecars']:
  p=Path(ref['path']);assert sha(p)==ref['sha256'];x=load(p);l=x['layer'];sidecar_hashes[p.name]=sha(p)
  assert x['method_lineage']['mit_upstream_commit']=='f79c9167690ca705e877ae4dc55a841d1aae1247'
  rows.setdefault(l,[]).extend(x['expert_reports'])
  for report in x['expert_reports']:sources[(l,report['expert'])]={'layer':l,'part':x['gpu'],'k2':x['source_parts']['k2'],'k3':x['source_parts']['k3']}
 assert set(rows)==set(range(3,45))
 for l,experts in rows.items():
  assert sorted(x['expert'] for x in experts)==list(range(288))
  for x in experts:
   assert len(x['slices'])==12 and x['hessian']['selected_natural_routes']>0
   assert math.isclose(x['measured_error_reduction'],x['k2_proxy_error_sum']-x['k3_proxy_error_sum'],rel_tol=1e-12)
  experts.sort(key=lambda x:(-x['measured_error_reduction'],x['expert']))
 e576=expert_plan(rows);whole_order=sorted(rows,key=lambda l:(-sum(x['measured_error_reduction'] for x in rows[l]),l));assert whole_order[:3]==[5,32,6]
 layouts={'expert576':e576,'whole3':{str(l):(list(range(288)) if l in whole_order[:3] else []) for l in sorted(rows)}}
 receipt=load(ROOT/'b12x-runtime/runtime-gates-r3.json');kv=receipt['kv_gib'];tokens=receipt['aggregate_kv_tokens'];headroom=kv*(1-262144/tokens)
 plans={}
 for label,ids in layouts.items():
  count=sum(map(len,ids.values()));delta=count*3145728;chosen={(int(l),e) for l,es in ids.items() for e in es};proxy=sum(x['measured_error_reduction'] for l,experts in rows.items() for x in experts if (l,x['expert']) in chosen)
  parts={}
  for key in sorted(chosen):
   s=sources[key];part_key=f"layer-{s['layer']:02d}-part-{s['part']}"
   parts.setdefault(part_key,{'layer':s['layer'],'part':s['part'],'source_k2':s['k2'],'source_k3':s['k3'],'selected_original_experts':[]})['selected_original_experts'].append(key[1])
  quant={'bits':2,'rank_stacked_tp':4}
  if label=='whole3':quant['layer_bits']={str(l):3 for l in whole_order[:3]}
  else:quant['expert_bits']={l:{str(e):3 for e in es} for l,es in ids.items() if es}
  plan={'schema':'glm53-unpruned-allocation-design-v1','state':'DESIGN_ONLY_NOT_BUILT','name':label,'source_k2_repo':'0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw','source_k2_revision':'35b4b580debe2a1510a2eb042d9d6d4cd7fb3a6b','source_k2_manifest_sha256':'501641b947fa56afc9ed098bdc034e59c2bd229d4fa1a1d7b80e3727f10c8ba3','paired_k3_status_sha256':'4a4402ab7b47f91145db841ce0105b54c3f7fcc3663a110fa712ceb85d5d1f0f','proxy_inventory_sha256':sha(invp),'sidecar_sha256':sidecar_hashes,'selection_uses_heldout_quality':False,'selection_policy':'largest marginal summed K2-minus-K3 conditional normalized reconstruction proxy in8-expert blocks; ties lowerlayer thenoriginalexpertID' if label=='expert576' else 'highest summed independent calibration K2-minus-K3 proxy across entire layers','block8_policy':'Conservative design convenience, NOT a proven fused ABI requirement.','k3_experts_by_layer':ids,'upgraded_experts':count,'experts_retained':12096,'removed_experts':0,'mixed_precision_layers':sum(0<len(es)<288 for es in ids.values()),'quantization_config_proposal':quant,'runtime_support':'existing staged whole-layer override' if label=='whole3' else 'NOT IMPLEMENTED: two homogeneous precision banks and scoped loader dispatch required','additional_tensor_bytes':delta,'projected_tensor_bytes':111274792824+delta,'native_protected_tensor_bytes':33835039608,'native_mtp_tensor_count':889,'native_mtp_experts':288,'all_original_router_rows_preserved':True,'calibration_proxy_reduction_sum':proxy,'required_source_parts':parts,'memory_estimate':{'runtime_receipt_sha256':sha(ROOT/'b12x-runtime/runtime-gates-r3.json'),'baseline_kv_gib_rounded':kv,'baseline_aggregate_tokens':tokens,'requested_context':262144,'proportional_extra_gib_above_context_floor':headroom,'projected_extra_reserve_after_weights_gib':headroom-delta/2**30,'additional_workspace_gib_unmeasured':True,'vision_context_and_MTP_admission_proven':False,'requires_existing_expert_only_FP8_MTP_runtime_policy':True},'quality_prediction':None,'quality_accepted':False,'runtime_accepted':False}
  plans[label]=plan;p=HERE/(label+'-design.json');assert not p.exists();p.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
 current=sum(x['delta_sum'] for x in inv['layers'] if x['layer'] in (5,32))
 out={'state':'CALIBRATION_ONLY_DESIGNS_FROZEN','whole_layer_ranking':whole_order,'current_whole2_proxy':current,'expert576_proxy':plans['expert576']['calibration_proxy_reduction_sum'],'proxy_advantage_percent':100*(plans['expert576']['calibration_proxy_reduction_sum']/current-1),'quantization_proxy_definition':'sum12 trace(E H E^T)/trace(W H W^T) per projection/rank; conditional capped natural routes, not end-to-end logit sensitivity','minimum_selected_natural_routes':min(x['hessian']['selected_natural_routes'] for experts in rows.values() for x in experts),'projected_extra_reserve_gib':{n:p['memory_estimate']['projected_extra_reserve_after_weights_gib'] for n,p in plans.items()},'generator_sha256':sha(Path(__file__)),'plan_sha256':{n:sha(HERE/(n+'-design.json')) for n in plans}}
 (HERE/'design-receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
