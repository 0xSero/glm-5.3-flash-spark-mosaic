"""Detached CPU-build terminal receipt collector; no model payload transfer."""
from pathlib import Path
import hashlib,json,subprocess,time
W=Path(__file__).resolve().parent;R='/home/valentine/glm53-single-spark-release-20260911/quality/unpruned-next-candidate/projection14'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
while True:
 r=subprocess.run(['ssh','-o','ConnectTimeout=10','spark-de5c','cat',R+'/BUILD_RESULT.json'],capture_output=True,text=True,timeout=20)
 if r.returncode==0:status=json.loads(r.stdout);break
 time.sleep(30)
assert status['state']=='CPU_BUILD_COMPLETE_HASH_BOUND',status
out=W/'build-receipts';out.mkdir(exist_ok=False)
for n in ['BUILD_RESULT.json','build-inspect.json','build.log','build-launch.json','DISK_ADMISSION.json','model/EXL3_MANIFEST.json','model/BUILD_STATUS.json','model/config.json','model/quantization_config.json','model/protected-tensor-closure.json']:
 p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(subprocess.check_output(['ssh','spark-de5c','cat',R+'/'+n]))
m=json.loads((out/'model/EXL3_MANIFEST.json').read_text());i=json.loads((out/'build-inspect.json').read_text())[0]
assert i['State']['ExitCode']==0 and not i['State']['OOMKilled']
assert sha(out/'model/EXL3_MANIFEST.json')==status['manifest_sha256'];assert m['selection_sha256']==sha(W/'selection.json');assert m['builder_sha256']==sha(W/'assemble_projection.py')
assert m['tensor_bytes']==115502651256 and len(m['files'])==133 and m['protected_tensor_bytes']==33835039608 and m['native_mtp_experts']==288
for n in ['config.json','quantization_config.json','protected-tensor-closure.json']:assert sha(out/'model'/n)==m['metadata_sha256'][n]
old=json.loads((W.parent/'projection/build-receipts/EXL3_MANIFEST.json').read_text());oldfiles={x['path']:x for x in old['files']}
for f in m['files']:
 if '/retained-' in f['path']:assert f['sha256']==oldfiles[f['path']]['sha256'] and f['bytes']==oldfiles[f['path']]['bytes']
closure=json.loads((out/'model/protected-tensor-closure.json').read_text());assert closure['tensor_count']==2482 and closure['native_mtp_tensor_count']==889
receipt={**status,'state':'STRUCTURAL_BUILD_RECEIPTS_VERIFIED_NOT_QUALITY_ACCEPTED','weight_file_bytes':m['weight_file_bytes'],'protected_tensor_bytes':m['protected_tensor_bytes'],'quality_accepted':False,'runtime_accepted':False,'receipt_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
(W/'FINAL_BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
