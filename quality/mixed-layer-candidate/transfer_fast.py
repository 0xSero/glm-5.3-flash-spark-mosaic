"""Omarchy: transfer only four immutable K3 files over wired2822/fabric bridge."""
from pathlib import Path
import subprocess,json,time
W=Path('/home/sero/glm53-single-spark-release-20260911/quality/mixed-layer-candidate')
ROOT='/home/valentine/glm53-single-spark-release-20260911/quality/mixed-layer-candidate'
start=time.time();names=['layer-05-part-0.safetensors','layer-05-part-1.safetensors','layer-32-part-0.safetensors','layer-32-part-1.safetensors']
src=subprocess.Popen(['tar','-C','/home/sero/glm53-tr3/release-tr3-3bpw/layers','-cf','-',*names],stdout=subprocess.PIPE)
dst=subprocess.Popen(['ssh','-o','BatchMode=yes','sero@spark-2822','ssh','-o','BatchMode=yes','valentine@spark-raila.internal','tar','-C',ROOT+'/k3-source-fast','-xf','-'],stdin=src.stdout);src.stdout.close()
a=src.wait();b=dst.wait();assert a==b==0
(W/'transfer-fast-status.json').write_text(json.dumps({'state':'FOUR_K3_FILES_TRANSFERRED_HASH_CHECK_PENDING','elapsed_seconds':time.time()-start,'files':names},indent=2)+'\n')
print('TRANSFER_FINISHED',time.time()-start,flush=True)
