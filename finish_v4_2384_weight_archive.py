"""Complete strict weight relocation after tar reports an unreadable HF cache file."""
import json,subprocess,shlex,time
from pathlib import Path
r=Path('/home/sero/work/glm53-single-spark-release-20260911');p=r/'retired-v4-2384-relocation.json'
proc=Path('/proc/1581602/cmdline')
while proc.exists() and b'archive_retired_v4_2384.py' in proc.read_bytes():time.sleep(10)
j=json.loads(p.read_text());assert j['state']=='SOURCE_HASHED',j['state']
log=(r/'archive-retired-v4-2384.log').read_text();assert '.cache/huggingface/trees/' in log and 'Permission denied' in log
h='''import hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
'''
def run(host,code):subprocess.run(['ssh','-o','BatchMode=yes',host,shlex.join(['python3','-c',code])],check=True)
items=j['files'];assert len(items)==48
run(j['backup_host'],h+f"p=Path({j['backup_path']!r});items={items!r}\nfor x in items:\n f=p/x['path'];assert f.stat().st_size==x['bytes'] and sha(f)==x['sha256'],x['path']\n(p/'BACKUP_VERIFIED.json').write_text(json.dumps({{'files':items,'scope':'all48modelweightfiles; oneunreadableHFtreecachemetadata excluded'}},indent=2))")
j.update(state='BACKUP_VERIFIED',non_weight_cache_copy_error='Unprivileged source could not read one HF tree cache JSON; all48 weight files independently hashed and verified in backup.');p.write_text(json.dumps(j,indent=2))
run(j['source_host'],h+f"p=Path({j['source']!r});items={items!r}\nfor x in items:\n f=p/x['path'];assert f.stat().st_size==x['bytes'] and sha(f)==x['sha256'];f.unlink()")
j.update(state='WEIGHTS_RELOCATED',finished_unix=time.time());p.write_text(json.dumps(j,indent=2))
run(j['source_host'],f"from pathlib import Path;Path({j['source']!r},'WEIGHTS_RELOCATED.json').write_text({json.dumps(j,indent=2)!r})")
print(json.dumps({'state':j['state'],'weight_bytes':sum(x['bytes'] for x in items)}),flush=True)
