"""Run on2822; archive retired2384 V4 with complete parity before removingweights."""
import json,shlex,subprocess,time
from pathlib import Path
src_host='sero@spark-raila.internal';dst_host='valentine@spark-raila.internal'
src='/home/sero/models/deepseek-ai/DeepSeek-V4-Flash-0731'
dst='/home/valentine/glm53-single-spark-release-20260911/retired-v4-2384-backup'
receipt=Path('/home/sero/work/glm53-single-spark-release-20260911/retired-v4-2384-relocation.json');assert not receipt.exists()
def command(host,args):return ['ssh','-o','BatchMode=yes',host,shlex.join(args)]
def run(host,code):return subprocess.check_output(command(host,['python3','-c',code]),text=True)
hasher='''import hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
'''
manifest=json.loads(run(src_host,hasher+f"p=Path({src!r});fs=sorted(p.glob('*.safetensors'));assert len(fs)==48;print(json.dumps([{{'path':f.name,'bytes':f.stat().st_size,'sha256':sha(f)}} for f in fs]))"))
proof={'state':'SOURCE_HASHED','source_host':src_host,'source':src,'backup_host':dst_host,'backup_path':dst,'files':manifest,'started_unix':time.time()};receipt.write_text(json.dumps(proof,indent=2))
run(dst_host,f"from pathlib import Path;import shutil;p=Path({dst!r});assert not p.exists();assert shutil.disk_usage(p.parent).free>{sum(x['bytes'] for x in manifest)}+70*2**30;p.mkdir()")
reader=subprocess.Popen(command(src_host,['tar','-C',src,'-cf','-','.']),stdout=subprocess.PIPE)
writer=subprocess.Popen(command(dst_host,['tar','--skip-old-files','-C',dst,'-xf','-']),stdin=reader.stdout)
reader.stdout.close();assert writer.wait()==0 and reader.wait()==0
run(dst_host,hasher+f"p=Path({dst!r});items={manifest!r}\nfor x in items:\n f=p/x['path'];assert f.stat().st_size==x['bytes'] and sha(f)==x['sha256'],x['path']\n(p/'BACKUP_VERIFIED.json').write_text(json.dumps({{'source':{src!r},'files':items}},indent=2))")
proof['state']='BACKUP_VERIFIED';receipt.write_text(json.dumps(proof,indent=2))
run(src_host,hasher+f"p=Path({src!r});items={manifest!r}\nfor x in items:\n f=p/x['path'];assert f.stat().st_size==x['bytes'] and sha(f)==x['sha256'];f.unlink()\n(p/'WEIGHTS_RELOCATED.json').write_text(json.dumps({proof!r},indent=2))")
proof.update(state='WEIGHTS_RELOCATED',finished_unix=time.time());receipt.write_text(json.dumps(proof,indent=2));print(json.dumps({'state':proof['state'],'bytes':sum(x['bytes'] for x in manifest)}),flush=True)
