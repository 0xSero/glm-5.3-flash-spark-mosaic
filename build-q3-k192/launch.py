"""de5c bounded CPU build; no GPU access or changes to queued quality."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
r=Path('/home/valentine/glm53-single-spark-release-20260911');work=r/'build-q3-k192'
assert shutil.disk_usage(r).free>=150*2**30
assert not (r/'q3-massmax-k192').exists()
assert hashlib.sha256((work/'builder.py').read_bytes()).hexdigest()=='b5cd69f87c106c161b1349707a3b4bdcf8cd70c504d615cac76d2443f929f734'
assert all(os.access(p,os.R_OK) for p in (r/'source-q3').rglob('*.safetensors'))
image='sha256:54acf16064726697ab079dfbd297c1c1c260cb28867f8287f5ae2ec3bbb0a382'
cmd=['docker','run','-d','--name','glm53-q3-massmax-k192-cpu-build','--network','none','--cpus','2',
     '--memory','8g','--memory-swap','8g','--user','1000:1000',
     '-v',str(r)+':/release','-v',str(r/'source-q3')+':/source:ro',
     '-v',str(work)+':/build:ro','--entrypoint','python3',image,
     '/build/builder.py','--source','/source','--output','/release/q3-massmax-k192',
     '--inventory','/release/source-inventory/filehash-inventory.json',
     '--candidates','/release/sealed-observations/candidates.json','--metric','massmax_domain','--keep','192']
cid=subprocess.check_output(cmd,text=True).strip()
inspect=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
h=inspect['HostConfig']
assert not h.get('DeviceRequests') and h['NanoCpus']==2000000000 and h['Memory']==8*2**30 and h['MemorySwap']==8*2**30
proof={'state':'CPU_BUILD_LAUNCHED_NOT_YET_COMPLETE','container_id':cid,'image':image,'command':cmd,
       'free_bytes_at_launch':shutil.disk_usage(r).free,'started_unix':time.time(),
       'builder_sha256':hashlib.sha256((work/'builder.py').read_bytes()).hexdigest(),
       'maps_sha256':hashlib.sha256((r/'sealed-observations/candidates.json').read_bytes()).hexdigest(),
       'resource_limits':{'cpus':2,'memory_bytes':8*2**30,'swap_extra_bytes':0,'gpu_devices':[]}}
(work/'launch.json').write_text(json.dumps(proof,indent=2)+'\n')
(work/'initial-inspect.json').write_text(json.dumps(inspect,indent=2)+'\n')
print(json.dumps(proof))
