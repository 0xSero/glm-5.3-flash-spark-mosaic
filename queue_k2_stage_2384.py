"""Run2822: fill newly freed2384 quality lane; noGPUlaunch."""
import json,time,subprocess,shlex
from pathlib import Path
r=Path('/home/sero/work/glm53-single-spark-release-20260911');host='sero@spark-raila.internal';dst='/home/sero/work/glm53-single-spark-release-20260911'
def remote(args):return ['ssh','-o','BatchMode=yes',host,shlex.join(args)]
while True:
 p=r/'retired-v4-2384-relocation.json'
 if p.exists() and json.loads(p.read_text()).get('state')=='WEIGHTS_RELOCATED':break
 time.sleep(15)
subprocess.run(remote(['python3','-c',f"from pathlib import Path;import shutil;p=Path({dst!r});p.mkdir(exist_ok=True,parents=True);assert shutil.disk_usage(p).free>135*2**30"]),check=True)
a=subprocess.Popen(['docker','save','sha256:ddc1b6cf8d89f1f0c0294a9a7a3c86d63cb7ad486d236013d75fb46914e2c576'],stdout=subprocess.PIPE)
b=subprocess.Popen(remote(['docker','load']),stdin=a.stdout);a.stdout.close();assert b.wait()==a.wait()==0
subprocess.run(remote(['mkdir',dst+'/source-k2']),check=True)
a=subprocess.Popen(['ssh','-o','BatchMode=yes','valentine@spark-raila.internal',shlex.join(['tar','-C','/home/valentine/flash-experimental-staging-20260906/model','-cf','-','.'])],stdout=subprocess.PIPE)
b=subprocess.Popen(remote(['tar','--skip-old-files','-C',dst+'/source-k2','-xf','-']),stdin=a.stdout);a.stdout.close();assert b.wait()==a.wait()==0
while not (r/'quality-k2-ready.json').exists():time.sleep(10)
a=subprocess.Popen(['tar','-C',str(r),'-cf','-','quality-k2','source-inventory-k2','bf16-normalized','quality-eval','observer-src'],stdout=subprocess.PIPE)
b=subprocess.Popen(remote(['tar','--skip-old-files','-C',dst,'-xf','-']),stdin=a.stdout);a.stdout.close();assert b.wait()==a.wait()==0
proof={'state':'COPIED_HASH_VALIDATION_PENDING','destination_host':host,'destination':dst,'finished_unix':time.time()};(r/'k2-2384-stage.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
