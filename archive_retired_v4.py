"""Run on2822: relocate retired V4 weights only after full backup hash parity."""
import hashlib,json,subprocess,shlex,time
from pathlib import Path
source=Path('/home/sero/models/DeepSeek-V4-Flash-180B')
host='valentine@spark-raila.internal'
target='/home/valentine/glm53-single-spark-release-20260911/retired-v4-2822-backup'
root=Path('/home/sero/work/glm53-single-spark-release-20260911')
receipt=root/'retired-v4-relocation.json'
assert not receipt.exists()
def ssh(code):return subprocess.check_output(['ssh','-o','BatchMode=yes',host,shlex.join(['python3','-c',code])],text=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
files=sorted(source.glob('*.safetensors'));assert len(files)==46
manifest=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
proof={'state':'SOURCE_HASHED','source':str(source),'backup_host':host,'backup_path':target,'files':manifest,'started_unix':time.time()}
receipt.write_text(json.dumps(proof,indent=2))
ssh(f"from pathlib import Path;import shutil;p=Path({target!r});assert not p.exists();assert shutil.disk_usage(p.parent).free>{sum(x['bytes'] for x in manifest)}+40*2**30;p.mkdir()")
reader=subprocess.Popen(['tar','-C',str(source),'-cf','-','.'],stdout=subprocess.PIPE)
writer=subprocess.Popen(['ssh','-o','BatchMode=yes',host,shlex.join(['tar','--skip-old-files','-C',target,'-xf','-'])],stdin=reader.stdout)
reader.stdout.close();assert writer.wait()==0 and reader.wait()==0
code=f'''import hashlib,json
from pathlib import Path
p=Path({target!r});items={manifest!r}
for x in items:
 f=p/x['path'];assert f.stat().st_size==x['bytes'];h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==x['sha256'],x['path']
(p/'BACKUP_VERIFIED.json').write_text(json.dumps({{'source':{str(source)!r},'files':items}},indent=2))
print('BACKUP_VERIFIED')
'''
assert ssh(code).strip()=='BACKUP_VERIFIED'
proof['state']='BACKUP_VERIFIED';receipt.write_text(json.dumps(proof,indent=2))
# Revalidate unchanged originals before unlinking ONLY the backed-up weight files.
for x in manifest:
 p=source/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];p.unlink()
proof.update(state='WEIGHTS_RELOCATED',finished_unix=time.time())
receipt.write_text(json.dumps(proof,indent=2));(source/'WEIGHTS_RELOCATED.json').write_text(json.dumps(proof,indent=2))
print(json.dumps({'state':proof['state'],'bytes':sum(x['bytes'] for x in manifest),'backup_path':target}),flush=True)
