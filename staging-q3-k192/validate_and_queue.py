"""2822 CPU-only candidate seal, frozen evaluator snapshot, then guarded queue."""
import hashlib,importlib.util,json,shutil,subprocess
from pathlib import Path
r=Path('/home/sero/work/glm53-single-spark-release-20260911');work=r/'staging-q3-k192'
candidate=r/'q3-massmax-k192';snapshot=work/'quality-frozen'
assert not snapshot.exists();shutil.copytree(r/'quality-k176-run1',snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
expected={'evaluate_pruned_quality.py':'066760759ed8596ab330a261a30c994c9f4f0a8976af970ed2cb8a29522325db',
'deps/evaluate_low_bpw_quality.py':'0c02949a77d28fe7c45786e964a1241cbba8d33b076335ca6a08a4fcac28729a',
'deps/glm53_exl3_tp4.py':'488b6f93183dde8c5c5a8248b46ccf22875e8619ddc0a3aa48115da34578a1a7',
'deps/release_route_capture_v2.py':'3c5f5558d2a248fd375944067f992619a43fb3ed982b6eb6b30570b7ac1b15b8'}
for name,digest in expected.items():assert sha(snapshot/name)==digest
spec=importlib.util.spec_from_file_location('frozen_quality',snapshot/'evaluate_pruned_quality.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
experts,_=m.index_contract(candidate);assert experts==192
pin=m.validate_candidate_files(candidate,192);assert pin=='98e72202d67acc158e908701d04b6d228de74b01d8c1c66d3aa0df6a998b1cbc'
receipt={'state':'FULL_PAYLOAD_HASH_PASS','manifest_sha256':pin,'keep':192,'snapshot_sha256':expected,
         'index_sha256':sha(candidate/'model.safetensors.index.json'),'config_sha256':sha(candidate/'config.json')}
(work/'stage-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
log=(work/'queue.log').open('ab')
p=subprocess.Popen(['python3','-u',str(work/'queue_quality.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(work/'queue-process.json').write_text(json.dumps({'queue_pid':p.pid},indent=2)+'\n')
print(json.dumps({'state':receipt['state'],'queue_pid':p.pid}))
