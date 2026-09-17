"""CPU-only immutable candidate validation and quality-code snapshot."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

r=Path('/release');q=Path('/queue');candidate=r/'k2-massmax-k256'
pin='7c7a2e47c7cde8996ac54c6f2ca5c3f7532b682d5d7296408c97a14e8950f73c'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
assert sha(candidate/'EXL3_MANIFEST.json')==pin
snapshot=q/'quality-frozen'
assert not snapshot.exists()
shutil.copytree(r/'quality',snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
expected={'evaluate_pruned_quality.py':'066760759ed8596ab330a261a30c994c9f4f0a8976af970ed2cb8a29522325db',
'deps/evaluate_low_bpw_quality.py':'0c02949a77d28fe7c45786e964a1241cbba8d33b076335ca6a08a4fcac28729a',
'deps/glm53_exl3_tp4.py':'488b6f93183dde8c5c5a8248b46ccf22875e8619ddc0a3aa48115da34578a1a7',
'deps/release_route_capture_v2.py':'3c5f5558d2a248fd375944067f992619a43fb3ed982b6eb6b30570b7ac1b15b8'}
for name,pinned in expected.items():assert sha(snapshot/name)==pinned, name
spec=importlib.util.spec_from_file_location('frozen_quality',snapshot/'evaluate_pruned_quality.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
experts,_=module.index_contract(candidate);assert experts==256
assert module.validate_candidate_files(candidate,experts)==pin
config=json.loads((candidate/'config.json').read_text())
assert config['quantization_config']['bits']==2 and config['native_mtp_n_routed_experts']==288
manifest=json.loads((candidate/'EXL3_MANIFEST.json').read_text())
assert manifest['target_bpw']==2.0 and manifest['tensor_bytes']==102659360376
receipt={'state':'FULL_PAYLOAD_HASH_PASS','candidate_manifest_sha256':pin,'weight_files':len(manifest['files']),
         'file_bytes':sum(x['bytes'] for x in manifest['files']),'tensor_bytes':manifest['tensor_bytes'],
         'snapshot_sha256':expected,'candidate_index_sha256':sha(candidate/'model.safetensors.index.json'),
         'candidate_config_sha256':sha(candidate/'config.json'),'gpu_used':False}
(q/'stage-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
