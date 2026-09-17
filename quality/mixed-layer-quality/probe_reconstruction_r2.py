"""Bounded real EXL3 K2/K3 reconstruction smoke; no serving or quality score."""
import argparse,json
from pathlib import Path
from mixed_contract import expected_bits,validate_trellis_shape,sha,MANIFEST_SHA
import evaluate_mixed_quality as adapter
p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert sha(a.artifact/'EXL3_MANIFEST.json')==MANIFEST_SHA
adapter.prepare_native_imports()
import torch
from safetensors import safe_open
from glm53_exl3_tp4 import _load_linear
weights=json.loads((a.artifact/'model.safetensors.index.json').read_text())['weight_map'];results=[]
torch.cuda.set_device(0)
with torch.inference_mode():
 for layer in (3,5,32,44):
  for rank in range(4):
   for projection in ('gate_proj','up_proj','down_proj'):
    prefix=f'model.language_model.layers.{layer}.mlp.experts.0.{projection}.rank{rank}'
    tensors={}
    for suffix in ('suh','svh','trellis','mcg'):
     key=prefix+'.'+suffix
     with safe_open(a.artifact/weights[key],framework='pt',device='cuda:0') as h:tensors[key]=h.get_tensor(key)
    class Reader:
     def get_tensor(self,key):return tensors[key]
    validate_trellis_shape(prefix,tensors[prefix+'.trellis'].shape)
    down=projection=='down_proj';module=_load_linear(Reader(),prefix,512 if down else 4096,4096 if down else 512,torch.bfloat16)
    assert module.K==expected_bits(prefix)
    raw=module.get_weight_tensor().T
    print(json.dumps({"prefix":prefix,"raw_shape":list(raw.shape),"raw_dtype":str(raw.dtype),"inferred_K":module.K}),flush=True)
    w=raw.to(dtype=torch.bfloat16)
    assert tuple(w.shape)==((4096,512) if down else (512,4096)) and w.dtype==torch.bfloat16 and torch.isfinite(w).all()
    results.append({'prefix':prefix,'inferred_K':module.K,'shape':list(w.shape),'dtype':str(w.dtype),'raw_dtype':str(raw.dtype)})
    del module,raw,w,tensors
 torch.cuda.synchronize()
report={'state':'REAL_EXL3_RECONSTRUCTION_PASS','candidate_manifest_sha256':MANIFEST_SHA,'projections':results,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'quality_measured':False}
assert not a.output.exists();a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'state':report['state'],'slices':len(results),'peak_allocated_bytes':report['peak_allocated_bytes']}))
