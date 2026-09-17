"""Run on2822: copy only missing Q3 shards across the fabric, preserving sources."""
import json,shlex,subprocess,time
from pathlib import Path
src_host='valentine@spark-raila.internal';dst_host='valentine@spark-raila.internal'
src='/home/valentine/glm53-single-spark-release-20260911/source-q3'
dst='/home/valentine/glm53-single-spark-release-20260911/source-q3'
def command(host,args):return ['ssh','-o','BatchMode=yes',host,shlex.join(args)]
def inventory(host,path):
 code=f"import json;from pathlib import Path;p=Path({path!r});print(json.dumps({{str(f.relative_to(p)):f.stat().st_size for f in p.rglob('*.safetensors') if '.cache' not in f.parts}}))"
 return json.loads(subprocess.check_output(command(host,['python3','-c',code]),text=True))
s=inventory(src_host,src);d=inventory(dst_host,dst)
assert all(s[n]==d[n] for n in s.keys()&d.keys())
names=sorted(s.keys()-d.keys());assert names
assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
total=sum(s[n] for n in names)
check=f"import shutil;assert shutil.disk_usage({dst!r}).free>{total}+20*2**30"
subprocess.run(command(dst_host,['python3','-c',check]),check=True)
r=Path('/home/sero/work/glm53-single-spark-release-20260911');r.mkdir(exist_ok=True)
receipt=r/'source-q3-control-transfer.json';assert not receipt.exists()
proof={'state':'RUNNING','source':src,'destination':dst,'files':names,'bytes':total,'started_unix':time.time(),'scope':'transfer only; full pinned HF hash validation follows'}
receipt.write_text(json.dumps(proof,indent=2))
with (r/'source-q3-control-transfer.log').open('w') as log:
 reader=subprocess.Popen(command(src_host,['tar','-C',src,'-cf','-',*names]),stdout=subprocess.PIPE,stderr=log)
 writer=subprocess.Popen(command(dst_host,['tar','--skip-old-files','-C',dst,'-xf','-']),stdin=reader.stdout,stdout=log,stderr=log)
 reader.stdout.close();right=writer.wait();left=reader.wait()
 proof.update(reader_exit_code=left,writer_exit_code=right,finished_unix=time.time())
 assert left==right==0,(left,right)
got=inventory(dst_host,dst)
assert all(got[n]==s[n] for n in names)
proof['state']='COPIED_SIZE_CHECKED';receipt.write_text(json.dumps(proof,indent=2));print(json.dumps({'state':proof['state'],'files':len(names),'bytes':total}),flush=True)
