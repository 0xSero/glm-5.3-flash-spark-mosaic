#!/usr/bin/env python3
"""Run de5c: gate one K2keep256 capture on completed/sealed Q3 and idle GPU."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/valentine/glm53-single-spark-release-20260911')
QUEUE=ROOT/'staging-k2-k256'
BASELINE='4f6816f2653e217ef0cd0a552cba7a9be5b2c4c7e7f5e3b864934af5e7aec3ef'
IMAGE='sha256:54acf16064726697ab079dfbd297c1c1c260cb28867f8287f5ae2ec3bbb0a382'
NAME='glm53-quality-k2-massmax-k256-attempt1'
PIN='7c7a2e47c7cde8996ac54c6f2ca5c3f7532b682d5d7296408c97a14e8950f73c'


def status(state,**fields):
 data={'state':state,'time':time.time(),**fields}
 (QUEUE/'queue-status.json.tmp').write_text(json.dumps(data,indent=2)+'\n')
 (QUEUE/'queue-status.json.tmp').replace(QUEUE/'queue-status.json')
 print(json.dumps(data),flush=True)


def inspect(cid):return json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
def cpu(command):
 return subprocess.check_output(['docker','run','--rm','--network','none','--memory','1g','--cpus','1',
  '-v',str(ROOT)+':/release:ro','-v',str(QUEUE)+':/queue','--entrypoint','python3',IMAGE]+command,text=True)


def main():
 with (QUEUE/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  run_reserved()

def run_reserved():
 receipt=json.loads((QUEUE/'stage-receipt.json').read_text())
 assert receipt['state']=='FULL_PAYLOAD_HASH_PASS' and receipt['candidate_manifest_sha256']==PIN
 if (QUEUE/'quality-launch.json').exists():raise RuntimeError('a launch receipt already exists; do not duplicate')
 while True:
  old=inspect(BASELINE)
  assert old['Id']==BASELINE and old['Image']==IMAGE
  if not old['State']['Running']:break
  status('WAITING_ORIGINAL_Q3_CONTROL',baseline_container_id=BASELINE)
  time.sleep(30)
 (QUEUE/'original-q3-terminal-inspect.json').write_text(json.dumps(old,indent=2)+'\n')
 with (QUEUE/'original-q3-terminal.log').open('wb') as log:
  subprocess.run(['docker','logs',BASELINE],stdout=log,stderr=subprocess.STDOUT,check=True)
 if old['State']['ExitCode']!=0:
  raise RuntimeError('original Q3 control failed; logs retained; no candidate GPU launch')
 status('SEALING_ORIGINAL_Q3_RESULTS')
 seal=cpu(['/queue/seal_baseline.py'])
 (QUEUE/'seal-baseline.stdout.txt').write_text(seal)
 cpu(['-c',"from pathlib import Path;assert not Path('/release/results/k2-massmax-k256').exists(), 'candidate output exists'"])
 # Do not launch into another owner's reserved/initializing GPU container.
 while True:
  pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  active=subprocess.check_output(['docker','ps','-q'],text=True).split()
  owners=[]
  for cid in active:
   item=inspect(cid)
   if any('gpu' in group for req in item['HostConfig'].get('DeviceRequests') or [] for group in req.get('Capabilities',[])):
    owners.append(item['Name'])
  if not pids and not owners:break
  status('WAITING_GPU_OWNER_CLEAR',pids=pids,containers=owners)
  time.sleep(30)
 # Verify frozen runnable files immediately before launch.
 for name,expected in receipt['snapshot_sha256'].items():
  assert hashlib.sha256((QUEUE/'quality-frozen'/name).read_bytes()).hexdigest()==expected
 cmd=['docker','run','-d','--gpus','all','--ipc','host','--network','none','--name',NAME,
      '-v',str(ROOT)+':/release','-v',str(QUEUE/'quality-frozen')+':/quality:ro',
      '-v','/home/valentine/glm53-full-observations-20260907/src:/workspace/src:ro',
      '-v',str(ROOT/'k2-massmax-k256')+':/candidate:ro',
      '-e','PYTHONPATH=/quality/deps:/workspace/src','-e','GLM53_EXL3_GPU_IDS=0',
      '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','PYTHONUNBUFFERED=1',
      IMAGE,'/quality/evaluate_pruned_quality.py','--phase','all','--artifact','/candidate',
      '--fixture','/release/quality-eval','--teacher','/release/bf16-normalized',
      '--output','/release/results/k2-massmax-k256']
 cid=subprocess.check_output(cmd,text=True).strip()
 proof={'state':'LAUNCHED_PENDING_VALIDATION','container_id':cid,'image':IMAGE,'command':cmd,
        'candidate_manifest_sha256':PIN,'stage_receipt_sha256':hashlib.sha256((QUEUE/'stage-receipt.json').read_bytes()).hexdigest(),
        'baseline_seal_sha256':hashlib.sha256((QUEUE/'original-q3-results-seal.json').read_bytes()).hexdigest(),
        'queue_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_unix':time.time()}
 (QUEUE/'quality-launch.json').write_text(json.dumps(proof,indent=2)+'\n')
 status('QUALITY_LAUNCHED',container_id=cid,candidate_manifest_sha256=PIN)
 # Preserve complete final log and inspect regardless of outcome.
 exit_code=subprocess.check_output(['docker','wait',cid],text=True).strip()
 final=inspect(cid)
 (QUEUE/'candidate-terminal-inspect.json').write_text(json.dumps(final,indent=2)+'\n')
 with (QUEUE/'candidate-terminal.log').open('wb') as log:
  subprocess.run(['docker','logs',cid],stdout=log,stderr=subprocess.STDOUT,check=True)
 if final['State']['ExitCode']!=0:
  raise RuntimeError('candidate quality failed; full terminal log/inspect retained')
 final_seal=cpu(['/queue/seal_baseline.py','candidate'])
 (QUEUE/'seal-candidate.stdout.txt').write_text(final_seal)
 status('CANDIDATE_QUALITY_COMPLETE_SEALED',exit_code=exit_code,container_id=cid)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  status('FAILED_PRESERVED_NO_RETRY',error=repr(exc))
  raise
