"""Prospective memory accounting; no serving admission or quality prediction."""
from pathlib import Path
import json,hashlib,math
w=Path(__file__).resolve().parent;r=w.parents[1]
adm=r/'b12x-runtime/mtp-baseline-replay/d2-c1-attempt2/admission.json';draft=r/'b12x-runtime/mtp-draft-nvfp4/FULL_MEMORY_COMPARISON.json';sel=r/'quality/unpruned-next-candidate/projection14/selection.json'
a=json.loads(adm.read_text());m=json.loads(draft.read_text());s=json.loads(sel.read_text())
assert a['max_num_seqs']==1 and a['context_limit']==262144 and a['kv_capacity_tokens']>2*a['context_limit']
assert m['resident_saving_bytes']==3170885120
assert s['extra_tensor_bytes']==4227858432 and s['extra_tensor_bytes']%14==0
per_layer=s['extra_tensor_bytes']//14
rows=[]
# Linear estimates are deliberately not treated as a real hybrid KV allocation.
for n in (14,18,20,22,24):
 extra=n*per_layer;net=extra-m['resident_saving_bytes'];remaining=a['kv_memory_gib']-net/2**30
 rows.append({'down_k3_layers':n,'extra_target_GiB':extra/2**30,'net_extra_after_measured_draft_saving_GiB':net/2**30,'hypothetical_remaining_KV_allowance_GiB':remaining,'linear_capacity_estimate_NOT_ADMITTED':int(remaining/a['kv_memory_gib']*a['kv_capacity_tokens'])})
result={'state':'PROSPECTIVE_ACCOUNTING_ONLY','actual_baseline_context_limit':a['context_limit'],'actual_baseline_slots':a['max_num_seqs'],'reported_KV_capacity':a['kv_capacity_tokens'],'reported_available_KV_GiB':a['kv_memory_gib'],'scope':'C1 currently admits one262144request despite larger aggregate KV capacity. More target precision might use some allowance, without reducing required context. Full-server NVFP4 admission first.','source_sha256':{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (adm,draft,sel)},'rows':rows,'not_proven':['Linear KV scaling ignores hybrid block rounding and fixed/per-sequence state.','Draft-only measured saving may not equal full-server retained saving.','Larger target transients, native MTP graphs, media and OS reserve must be measured.','No predicted fidelity score, no selected release, no lower context or quantized protected target tensors.']}
(w/'ASSESSMENT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))
