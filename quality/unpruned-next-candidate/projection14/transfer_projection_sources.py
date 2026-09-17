"""Omarchy: transfer only eight new layers of immutable K3 files over wired2822/fabric bridge."""
from pathlib import Path
import subprocess,json,time
W=Path('/home/sero/glm53-single-spark-release-20260911/quality/unpruned-next-candidate/projection14')
ROOT='/home/valentine/glm53-single-spark-release-20260911/quality/unpruned-next-candidate/projection14'
start=time.time();names=[f'layer-{l:02d}-part-{p}.safetensors' for l in (6,7,20,22,23,27,28,37) for p in (0,1)]
src=subprocess.Popen(['tar','-C','/home/sero/glm53-tr3/release-tr3-3bpw/layers','-cf','-',*names],stdout=subprocess.PIPE)
dst=subprocess.Popen(['ssh','-o','BatchMode=yes','sero@spark-2822','ssh','-o','BatchMode=yes','valentine@spark-raila.internal','tar','-C',ROOT+'/k3-source','-xf','-'],stdin=src.stdout);src.stdout.close()
a=src.wait();b=dst.wait();assert a==b==0
(W/'transfer-fast-status.json').write_text(json.dumps({'state':'SIXTEEN_K3_FILES_TRANSFERRED_HASH_CHECK_PENDING','elapsed_seconds':time.time()-start,'files':names},indent=2)+'\n')
print('TRANSFER_FINISHED',time.time()-start,flush=True)
