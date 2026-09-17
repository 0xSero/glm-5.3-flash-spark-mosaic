#!/usr/bin/env python3
"""de5c CPU-only staging supervisor; never allocates GPUs or touches live servers."""
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path('/home/valentine/glm53-single-spark-release-20260911');WORK=ROOT/'quality/mixed-layer-candidate'
IMAGE='sha256:54acf16064726697ab079dfbd297c1c1c260cb28867f8287f5ae2ec3bbb0a382'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(name,**kw):
 d={'state':name,'time':time.time(),**kw};p=WORK/'stage-status.json';p.with_suffix('.tmp').write_text(json.dumps(d,indent=2)+'\n');p.with_suffix('.tmp').replace(p);print(json.dumps(d),flush=True)
def main():
 assert not (WORK/'launch.json').exists() and not (WORK/'model').exists()
 q=json.loads((WORK/'paired-k3-availability.json').read_text());selected=[x for x in q['parts'] if x['layer'] in [5,32]]
 while True:
  pending=[]
  for x in selected:
   p=WORK/'k3-source'/Path(x['weight_path']).name
   if not p.exists() or p.stat().st_size!=x['declared_bytes']:pending.append(p.name)
  if not pending:break
  state('WAITING_FOUR_K3_PARTS',pending=pending);time.sleep(10)
 source={};native=ROOT/'source-q3';old=json.loads((native/'EXL3_MANIFEST.json').read_text())
 for x in old['files']:source[x['sha256']]=str(native/x['path'])
 cache=Path('/home/valentine/glm53-exl3-staging/release-trellis-k2-2bpw/layers')
 for p in cache.glob('layer-*-part-*.json'):
  x=json.loads(p.read_text());source[x['sha256']]=str(p.with_suffix('.safetensors'))
 for x in selected:source[x['declared_sha256']]=str(WORK/'k3-source'/Path(x['weight_path']).name)
 inv=json.loads((WORK/'source-metadata/filehash-inventory.json').read_text())
 for x in inv['files']:
  p=Path(source[x['sha256']]);assert p.is_file() and p.stat().st_size==x['bytes']
 (WORK/'source-map.json').write_text(json.dumps(source,indent=2)+'\n')
 state('STARTING_BOUNDED_CPU_BUILD')
 cmd=['docker','run','-d','--name','glm53-mixed-k2-k3-l5-l32-cpu-build','--network','none','--cpus','2','--memory','8g','--memory-swap','8g','--user','1000:1000','-v','/home/valentine:/home/valentine','--entrypoint','python3',IMAGE,'-u',str(WORK/'assemble.py'),'--source-map',str(WORK/'source-map.json'),'--output',str(WORK/'model')]
 cid=subprocess.check_output(cmd,text=True).strip();(WORK/'launch.json').write_text(json.dumps({'container_id':cid,'command':cmd,'builder_sha256':sha(WORK/'assemble.py'),'source_map_sha256':sha(WORK/'source-map.json'),'gpu_devices':[],'source_policy':'Known source paths are opened only for read; hardlinks change link counts only. Candidate metadata writes are isolated under new model directory.'},indent=2)+'\n')
 state('CPU_BUILD_RUNNING',container_id=cid)
 code=subprocess.check_output(['docker','wait',cid],text=True).strip()
 raw=subprocess.check_output(['docker','inspect',cid]);(WORK/'final-inspect.json').write_bytes(raw)
 with (WORK/'build.log').open('wb') as f:subprocess.run(['docker','logs',cid],stdout=f,stderr=subprocess.STDOUT,check=True)
 info=json.loads(raw)[0];assert info['State']['ExitCode']==0 and not info['State']['OOMKilled']
 complete=json.loads((WORK/'model/BUILD_STATUS.json').read_text());assert complete['state']=='COMPLETE'
 state('ASSEMBLY_COMPLETE_NO_GPU_QUALITY',container_id=cid,manifest_sha256=complete['manifest_sha256'],tensor_bytes=complete['tensor_bytes'])
if __name__=='__main__':
 try:main()
 except Exception as e:state('FAILED_PRESERVED',error=repr(e));raise
