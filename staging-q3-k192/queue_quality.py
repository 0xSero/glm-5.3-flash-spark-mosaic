"""2822: launch192 only with freeGPU, fresh>=35GiBMemAvailable,>=20GiBdisk."""
import fcntl,hashlib,json,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/sero/work/glm53-single-spark-release-20260911');WORK=ROOT/'staging-q3-k192'
IMAGE='sha256:ddc1b6cf8d89f1f0c0294a9a7a3c86d63cb7ad486d236013d75fb46914e2c576'
PIN='98e72202d67acc158e908701d04b6d228de74b01d8c1c66d3aa0df6a998b1cbc'

def status(state,**kw):
 value={'state':state,'time':time.time(),**kw};p=WORK/'queue-status.json'
 p.with_suffix('.tmp').write_text(json.dumps(value,indent=2)+'\n');p.with_suffix('.tmp').replace(p)
 print(json.dumps(value),flush=True)
def memory_available():
 return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
def inspect(cid):return json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
def cpu(args):
 return subprocess.check_output(['docker','run','--rm','--network','none','--memory','1g','--cpus','1',
  '-v',str(ROOT)+':/release:ro','-v',str(WORK)+':/queue','--entrypoint','python3',IMAGE]+args,text=True)

def execute():
 receipt=json.loads((WORK/'stage-receipt.json').read_text());assert receipt['manifest_sha256']==PIN
 assert not (WORK/'launch.json').exists()
 cpu(['-c',"from pathlib import Path;assert not Path('/release/results/massmax-k192').exists()"])
 while True:
  pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  owners=[]
  for cid in subprocess.check_output(['docker','ps','-q'],text=True).split():
   item=inspect(cid)
   if any('gpu' in group for req in item['HostConfig'].get('DeviceRequests') or [] for group in req.get('Capabilities',[])):owners.append(item['Name'])
  available=memory_available();free=shutil.disk_usage(ROOT).free
  hold=(WORK/'HOLD_GPU_LAUNCH').exists()
  if not hold and not pids and not owners and available>=35*2**30 and free>=20*2**30:break
  status('WAITING_RESOURCE_ADMISSION',explicit_gpu_hold=hold,mem_available_bytes=available,disk_free_bytes=free,gpu_pids=pids,gpu_owners=owners)
  time.sleep(30)
 for name,expected in receipt['snapshot_sha256'].items():assert hashlib.sha256((WORK/'quality-frozen'/name).read_bytes()).hexdigest()==expected
 cmd=['docker','run','-d','--name','glm53-quality-q3-k192-attempt1','--network','none','--ipc','host','--gpus','all',
      '-v',str(WORK/'quality-frozen')+':/quality:ro','-v',str(ROOT/'observer-src')+':/workspace/src:ro',
      '-v',str(ROOT/'q3-massmax-k192')+':/candidate:ro','-v',str(ROOT)+':/release',
      '-e','PYTHONPATH=/quality:/quality/deps:/workspace/src','-e','GLM53_EXL3_GPU_IDS=0',
      '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','PYTHONUNBUFFERED=1',IMAGE,
      '/quality/evaluate_pruned_quality.py','--phase','all','--artifact','/candidate',
      '--fixture','/release/quality-eval','--teacher','/release/bf16-normalized','--output','/release/results/massmax-k192']
 cid=subprocess.check_output(cmd,text=True).strip()
 proof={'state':'LAUNCHED_PENDING_QUALITY','container_id':cid,'image':IMAGE,'command':cmd,'candidate_manifest_sha256':PIN,
        'mem_available_before_launch':available,'disk_free_before_launch':free,'snapshot_sha256':receipt['snapshot_sha256'],
        'queue_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_unix':time.time()}
 (WORK/'launch.json').write_text(json.dumps(proof,indent=2)+'\n');status('QUALITY_LAUNCHED',container_id=cid)
 code=subprocess.check_output(['docker','wait',cid],text=True).strip();final=inspect(cid)
 (WORK/'final-inspect.json').write_text(json.dumps(final,indent=2)+'\n')
 with (WORK/'quality.log').open('wb') as log:subprocess.run(['docker','logs',cid],stdout=log,stderr=subprocess.STDOUT,check=True)
 if final['State']['ExitCode']!=0:raise RuntimeError('quality failed; final log/inspect preserved')
 output=cpu(['/queue/seal_results.py']);(WORK/'seal.stdout.txt').write_text(output)
 status('QUALITY_COMPLETE_SEALED',container_id=cid,exit_code=code)

def main():
 with (WORK/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute()

if __name__=='__main__':
 try:main()
 except Exception as exc:status('FAILED_PRESERVED',error=repr(exc));raise
