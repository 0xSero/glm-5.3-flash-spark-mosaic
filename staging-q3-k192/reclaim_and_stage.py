#!/usr/bin/env python3
"""2822: prove557f backup, remove only temp176 weights, stage192 and verify."""
import fcntl,hashlib,json,shlex,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/sero/work/glm53-single-spark-release-20260911')
WORK=ROOT/'staging-q3-k192'
OLD_PIN='0230ee28b89f0eab7b24bebca8ddd10ff3e96783f408efe17ca1a9bbe0f39a13'
NEW_PIN='98e72202d67acc158e908701d04b6d228de74b01d8c1c66d3aa0df6a998b1cbc'
SOURCE='/home/valentine/glm53-single-spark-release-20260911/q3-massmax-k192'

def remote(host,args):return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(args)]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def status(state,**kw):
 value={'state':state,'time':time.time(),**kw};p=WORK/'status.json'
 p.with_suffix('.tmp').write_text(json.dumps(value,indent=2)+'\n');p.with_suffix('.tmp').replace(p)
 print(json.dumps(value),flush=True)

def main():
 with (WORK/'job.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  execute()

def execute():
 status('VERIFYING_PRESERVED_557F_KEEP176')
 output=subprocess.check_output(remote('valentine@spark-raila.internal',['python3','-']),input=(WORK/'verify_preserved_copy.py').read_bytes())
 backup=json.loads(output);assert backup['state']=='PRESERVED_COPY_FULL_HASH_PASS' and backup['manifest_sha256']==OLD_PIN
 (WORK/'preserved-copy-proof.json').write_bytes(output)
 old=ROOT/'q3-massmax-k176';assert sha(old/'EXL3_MANIFEST.json')==OLD_PIN
 manifest=json.loads((old/'EXL3_MANIFEST.json').read_text())
 expected=[{k:x[k] for k in ('path','bytes','sha256')} for x in manifest['files'] if not x.get('omitted_empty_shard')]
 assert expected==backup['files']
 # Whole quality output tree remains outside this strictly bounded deletion set.
 delete=[]
 for item in expected:
  path=old/item['path']
  assert path.suffix=='.safetensors' and not path.is_symlink() and old in path.resolve().parents
  assert path.is_file() and path.stat().st_size==item['bytes']
  delete.append(path)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 # This failed candidate's GPU job must be terminal; never evict an active run.
 for cid in subprocess.check_output(['docker','ps','-q'],text=True).split():
  active=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
  assert not any(str(old) in str(x) for x in active['Config'].get('Cmd') or []),active['Name']
  assert not any(m.get('Source')==str(old) for m in active.get('Mounts',[])),active['Name']
 before=shutil.disk_usage(ROOT).free
 status('RECLAIMING_ONLY_TEMP_KEEP176_WEIGHTS',files=len(delete),backup_manifest_sha256=OLD_PIN)
 for path in delete:path.unlink()
 proof={'state':'TEMP_KEEP176_WEIGHTS_REMOVED_BACKUP_VERIFIED','removed_files':expected,
        'backup_proof_sha256':sha(WORK/'preserved-copy-proof.json'),'free_before':before,
        'free_after':shutil.disk_usage(ROOT).free,'results_preserved':str(ROOT/'results/massmax-k176')}
 (WORK/'reclamation.json').write_text(json.dumps(proof,indent=2)+'\n')
 assert shutil.disk_usage(ROOT).free>=110847041992+20*2**30
 target=ROOT/'q3-massmax-k192';assert not target.exists();target.mkdir()
 check="from pathlib import Path;import hashlib,json;r=Path("+repr(SOURCE)+");assert hashlib.sha256((r/'EXL3_MANIFEST.json').read_bytes()).hexdigest()=="+repr(NEW_PIN)+";assert json.loads((r/'BUILD_STATUS.json').read_text())['state']=='COMPLETE'"
 subprocess.run(remote('valentine@spark-raila.internal',['python3','-c',check]),check=True)
 status('STAGING_Q3_KEEP192',free_bytes=shutil.disk_usage(ROOT).free)
 source=subprocess.Popen(remote('valentine@spark-raila.internal',['tar','-C',SOURCE,'-cf','-','.']),stdout=subprocess.PIPE)
 dest=subprocess.Popen(['tar','-C',str(target),'-xf','-'],stdin=source.stdout);source.stdout.close()
 dest_rc=dest.wait();source_rc=source.wait();assert source_rc==dest_rc==0
 assert sha(target/'EXL3_MANIFEST.json')==NEW_PIN
 status('COPIED_VALIDATING_KEEP192_HASHES')
 # Existing2822 quality snapshot/evaluator is reused; no changes to any running code.
 subprocess.run(['python3',str(WORK/'validate_and_queue.py')],check=True)
 status('QUALITY_QUEUE_STARTED')

if __name__=='__main__':
 try:main()
 except Exception as exc:
  status('FAILED_PRESERVED',error=repr(exc));raise
