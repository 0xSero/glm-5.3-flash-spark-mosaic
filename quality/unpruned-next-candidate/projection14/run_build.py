"""CPU-only sealed projection14 assembly, preserving terminal failure evidence."""
from pathlib import Path
import subprocess,json,hashlib,shutil
W=Path(__file__).resolve().parent
assert not (W/'build-launch.json').exists()
s=json.loads((W/'selection.json').read_text());old={5,26,31,32,35,36}
for x in s['selected_source_parts']:
 if x['layer'] not in old:
  p=W/'k3-source'/f"layer-{x['layer']:02d}-part-{x['part']}.safetensors"
  assert p.stat().st_size==x['k3']['bytes'],'incomplete transfer'
assert shutil.disk_usage(W).free>10*2**30
cmd=['docker','run','-d','--name','glm53-projection14-cpu-build','--network','none','--cpus','2','--memory','8g','--memory-swap','8g','--user','1000:1000','-v','/home/valentine:/home/valentine','sha256:54acf16064726697ab079dfbd297c1c1c260cb28867f8287f5ae2ec3bbb0a382',str(W/'assemble_projection.py')]
cid=subprocess.check_output(cmd,text=True).strip();(W/'build-launch.json').write_text(json.dumps({'cid':cid,'command':cmd,'builder_sha256':hashlib.sha256((W/'assemble_projection.py').read_bytes()).hexdigest()},indent=2)+'\n')
subprocess.check_output(['docker','wait',cid]);raw=subprocess.check_output(['docker','inspect',cid]);(W/'build-inspect.json').write_bytes(raw)
with (W/'build.log').open('wb') as f:subprocess.run(['docker','logs',cid],stdout=f,stderr=subprocess.STDOUT,check=True)
i=json.loads(raw)[0]
if i['State']['ExitCode']!=0 or i['State']['OOMKilled']:
 (W/'BUILD_RESULT.json').write_text(json.dumps({'state':'FAILED_PRESERVED','cid':cid})+'\n');raise RuntimeError('build failed, evidence retained')
status=json.loads((W/'model/BUILD_STATUS.json').read_text());assert status['state']=='COMPLETE'
h=hashlib.sha256((W/'model/EXL3_MANIFEST.json').read_bytes()).hexdigest();assert status['manifest_sha256']==h
(W/'BUILD_RESULT.json').write_text(json.dumps({**status,'state':'CPU_BUILD_COMPLETE_HASH_BOUND','cid':cid},indent=2)+'\n')
