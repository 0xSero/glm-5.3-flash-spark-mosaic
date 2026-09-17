"""CPU-only prospective budget from pinned source metadata/calibration; no quality labels."""
import hashlib,json,re
from pathlib import Path
W=Path(__file__).resolve().parent;P=W.parent/'projection';G=2**30
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def build():
 s=json.loads((P/'selection.json').read_text())
 assert sha(P/'selection.json')=='38efe0f4da5d382499d4a3fb220ac66c8597b59190e11cd7679038f21afc6567'
 for n,h in s['projection_input_sha256'].items():assert sha(P/n)==h
 native=json.loads((W/'native-mtp-header-inventory.json').read_text());assert native['manifest_sha256']=='590ec08e84a66767e03964f7482e84c952e6b98c26cddc7d45983592c4e491f5'
 ts=native['tensors'];experts={k:v for k,v in ts.items() if re.search(r'\.mlp\.experts\.\d+\.',k)}
 assert len(ts)==889 and len(experts)==864 and all(v['dtype']=='BF16' and sorted(v['shape'])==[2048,4096] for v in experts.values())
 bf16=sum(v['bytes'] for v in experts.values());n=bf16//2
 fp8=n+2*288*4;nvfp4=n//2+n//16+4*288*4;saving=fp8-nvfp4
 ranking=sorted(s['rankings'],key=lambda r:(-r['paired_proxy_reduction']['down_proj'],r['layer']))
 # Verify the predeclared scalar ranking from the independent per-part sums.
 raw=[]
 for fname in ['k2-projection-errors.json','k3-projection-errors.json']:
  sums={l:0. for l in range(3,45)};doc=json.loads((P/fname).read_text())
  assert len(doc['parts'])==84
  for part in doc['parts']:
   for expert in part['reports']:sums[part['layer']]+=expert['proxy']['down_proj']
  raw.append(sums)
 for row in ranking:assert abs(raw[0][row['layer']]-raw[1][row['layer']]-row['paired_proxy_reduction']['down_proj'])<1e-9
 candidates=[]
 for count in [6,12,14,18,20]:
  selected=ranking[:count];extra=count*288*2**20
  candidates.append({'down_layers':count,'selected_layers':sorted(r['layer'] for r in selected),'layer_projection_bits':{str(r['layer']):{'down_proj':3} for r in selected},'extra_target_tensor_bytes':extra,'total_native_mtp_source_artifact_tensor_bytes':111274792824+extra,'net_payload_increase_vs_original_k2_fp8_draft':extra-saving,'calibration_proxy_reduction':sum(r['paired_proxy_reduction']['down_proj'] for r in selected),'predicted_agreement':None,'runtime_admitted':False})
 return {'state':'PROSPECTIVE_CPU_BUDGET_NOT_RUNTIME_ADMISSION','source_pins':{'selection_sha256':sha(P/'selection.json'),'native_mtp_inventory_sha256':sha(W/'native-mtp-header-inventory.json'),'paired_proxy_inventory_sha256':s['paired_proxy_inventory_sha256'],'input_sha256':s['projection_input_sha256']},'mtp':{'native_total_bytes':sum(v['bytes'] for v in ts.values()),'native_routed_bytes':bf16,'native_nonexpert_bytes':sum(v['bytes'] for v in ts.values())-bf16,'measured_fp8_parameter_bytes':fp8,'nominal_nvfp4_parameter_bytes':nvfp4,'nominal_saved_bytes':saving,'excluded':'backend padding/reordering, workspace, graph buffers, allocator fragmentation, quantization transient peaks'},'candidates':candidates,'recommendation':'Projection14 is the next prospective design if measured full-draft NVFP4 resident savings are at least2.75GiB and sufficient OS/KV reserve remains; otherwise projection12. No new artifact build or GPU measurement authorized by this script. Layer order is solely independent calibration; heldout quality is not read.'}
if __name__=='__main__':
 r=build();(W/'assessment.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'mtp':r['mtp'],'candidates':[{k:v for k,v in x.items() if k!='layer_projection_bits'} for x in r['candidates']]},indent=2))
