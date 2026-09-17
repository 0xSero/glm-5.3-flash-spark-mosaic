"""Run2822: stage sealed GLM candidate after verified retired-model relocation."""
import json,time,subprocess,shlex,shutil
from pathlib import Path
r=Path('/home/sero/work/glm53-single-spark-release-20260911')
while True:
 p=r/'retired-v4-relocation.json'
 if p.exists() and json.loads(p.read_text()).get('state')=='WEIGHTS_RELOCATED':break
 time.sleep(15)
src='/home/valentine/glm53-single-spark-release-20260911/q3-massmax-k176'
dst=r/'q3-massmax-k176';assert not dst.exists();assert shutil.disk_usage(r).free>125*2**30;dst.mkdir()
reader=subprocess.Popen(['ssh','-o','BatchMode=yes','valentine@spark-raila.internal',shlex.join(['tar','-C',src,'-cf','-','.'])],stdout=subprocess.PIPE)
writer=subprocess.Popen(['tar','--skip-old-files','-C',str(dst),'-xf','-'],stdin=reader.stdout)
reader.stdout.close();assert writer.wait()==reader.wait()==0
j=json.loads((dst/'BUILD_STATUS.json').read_text());assert j['state']=='COMPLETE'
(r/'q3-k176-stage.json').write_text(json.dumps({'state':'COPIED_HASH_VALIDATION_PENDING','source':src,'destination':str(dst),'structural_manifest_sha256':j['manifest_sha256'],'finished_unix':time.time()},indent=2));print('COPIED_HASH_VALIDATION_PENDING',flush=True)
