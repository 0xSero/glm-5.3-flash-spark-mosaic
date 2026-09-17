#!/usr/bin/env python3
"""Sequential bounded preflight, real reconstruction, then frozen full capture."""
import json,subprocess,time,hashlib
from pathlib import Path
ROOT=Path('/home/valentine/glm53-single-spark-release-20260911');W=ROOT/'quality/mixed-layer-quality';IMAGE='sha256:54acf16064726697ab079dfbd297c1c1c260cb28867f8287f5ae2ec3bbb0a382'
def state(s,**kw):
 p=W/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'state':s,'time':time.time(),**kw},indent=2)+'\n');tmp.replace(p)
def idle():
 p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
 if p.strip():raise RuntimeError('GPU owner appeared; preserve all other work')
def run(stage,gpu,script,args):
 if gpu:idle()
 cmd=['docker','run','-d','--network','none','--cpus','4','--memory',('64g' if gpu else '8g'),'--memory-swap',('64g' if gpu else '8g'),'--name','glm53-mixed-l5-l32-'+stage]
 if gpu:cmd+=['--gpus','all','--ipc','host']
 cmd+=['-v',str(ROOT)+':/release','-v',str(W)+':/quality:ro','-v','/home/valentine/glm53-full-observations-20260907/src:/workspace/src:ro','-v',str(ROOT/'quality/mixed-layer-candidate/model')+':/candidate:ro']
 for val in ['PYTHONPATH=/quality/deps:/workspace/src','GLM53_EXL3_GPU_IDS=0','OMP_NUM_THREADS=2','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','PYTHONUNBUFFERED=1']:cmd+=['-e',val]
 cmd += [IMAGE,'/quality/'+script,*args]
 cid=subprocess.check_output(cmd,text=True).strip();(W/(stage+'-launch.json')).write_text(json.dumps({'cid':cid,'command':cmd,'started':time.time()},indent=2)+'\n');state(stage.upper()+'_RUNNING',cid=cid)
 if stage=='capture':
  with (W/'monitor.log').open('ab') as log:
   mon=subprocess.Popen(['python3',str(W/'monitor_capture.py'),'--container',cid,'--result-root','/release/results/mixed-k2-k3-l5-l32','--output',str(W/'layer-history.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   (W/'monitor-process.json').write_text(json.dumps({'pid':mon.pid,'cid':cid})+'\n')
 subprocess.check_output(['docker','wait',cid]);raw=subprocess.check_output(['docker','inspect',cid]);(W/(stage+'-inspect.json')).write_bytes(raw)
 with (W/(stage+'.log')).open('wb') as log:subprocess.run(['docker','logs',cid],stdout=log,stderr=subprocess.STDOUT,check=True)
 info=json.loads(raw)[0]
 if info['State']['ExitCode']!=0 or info['State']['OOMKilled']:raise RuntimeError(stage+' failed; logs retained')
 return cid
def main():
 assert not(W/'status.json').exists();assert not(ROOT/'results/mixed-k2-k3-l5-l32').exists()
 idle();mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 assert mem>=50*2**30
 common=['--artifact','/candidate','--fixture','/release/quality-eval','--teacher','/release/bf16-normalized','--output','/release/results/mixed-k2-k3-l5-l32']
 run('preflight',False,'evaluate_mixed_quality.py',['--phase','preflight',*common])
 run('reconstruct',True,'probe_reconstruction.py',['--artifact','/candidate','--output','/release/quality/mixed-layer-quality/reconstruction-proof.json'])
 cid=run('capture',True,'evaluate_mixed_quality.py',['--phase','all',*common])
 state('CAPTURE_COMPLETE_PENDING_INTEGRITY_SEAL',cid=cid)
if __name__=='__main__':
 try:main()
 except Exception as e:state('FAILED_PRESERVED',error=repr(e));raise
