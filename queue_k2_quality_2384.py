"""Run2822: start one frozen K2 quality capture after verified staging completes."""
import json,time,subprocess,shlex
from pathlib import Path
r=Path('/home/sero/work/glm53-single-spark-release-20260911');host='sero@spark-raila.internal';rr='/home/sero/work/glm53-single-spark-release-20260911'
while True:
 p=r/'k2-2384-stage.json'
 if p.exists() and json.loads(p.read_text()).get('state')=='COPIED_HASH_VALIDATION_PENDING':break
 time.sleep(15)
code=f'''import json,shutil,subprocess,hashlib,time
from pathlib import Path
r=Path({rr!r});snap=r/'quality-k2-run1';assert not snap.exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
shutil.copytree(r/'quality-k2',snap)
image='sha256:ddc1b6cf8d89f1f0c0294a9a7a3c86d63cb7ad486d236013d75fb46914e2c576'
cmd=['docker','run','-d','--gpus','all','--ipc','host','--network','none','--name','glm53-quality-k2-control-attempt1','-v',str(r)+':/release','-v',str(snap)+':/quality:ro','-v',str(r/'observer-src')+':/workspace/src:ro','-v',str(r/'source-k2')+':/candidate:ro','-e','PYTHONPATH=/quality/deps:/workspace/src','-e','GLM53_EXL3_GPU_IDS=0','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','PYTHONUNBUFFERED=1',image,'/quality/evaluate_k2_quality.py','--phase','all','--source-inventory','/release/source-inventory-k2/filehash-inventory.json','--artifact','/candidate','--fixture','/release/quality-eval','--teacher','/release/bf16-normalized','--output','/release/results/original-k2-control']
cid=subprocess.check_output(cmd,text=True).strip()
proof={{'state':'LAUNCHED_PENDING_VALIDATION','container_id':cid,'command':cmd,'image':image,'snapshot_sha256':hashlib.sha256((snap/'evaluate_k2_quality.py').read_bytes()).hexdigest(),'started_unix':time.time()}}
(r/'k2-quality-attempt1-launch.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''
out=subprocess.check_output(['ssh','-o','BatchMode=yes',host,shlex.join(['python3','-c',code])],text=True)
(r/'k2-2384-quality-launch.json').write_text(out);print(out,flush=True)
