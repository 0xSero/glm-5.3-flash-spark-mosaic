"""Bounded projection6 preflight/capture; never changes existing sources/jobs."""
import argparse,json,subprocess,time,shutil
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[3];MODEL=W.parent/'model';IMAGE='sha256:54acf16064726697ab079dfbd297c1c1c260cb28867f8287f5ae2ec3bbb0a382';RESULT='/release/results/projection6-k2-gateup-k3-down'
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','capture'],required=True);a=p.parse_args();gpu=a.phase=='capture'
if gpu:
 proof=json.loads((W/'GPU_MICROPROOF.json').read_text());assert proof['state']=='VERIFIED_RUNTIME_PROJECTION_SMOKE'
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU owned'
 mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;assert mem>=50*2**30
assert json.loads((MODEL/'BUILD_STATUS.json').read_text())['state']=='COMPLETE'
assert shutil.disk_usage(W).free>=20*2**30
launch=W/(a.phase+'-launch.json');assert not launch.exists()
cmd=['docker','run','-d','--name','glm53-projection6-quality-'+a.phase,'--network','none','--cpus','4','--memory','64g' if gpu else '8g','--memory-swap','64g' if gpu else '8g']
if gpu:cmd+=['--gpus','all','--ipc','host']
cmd+=['-v',str(ROOT)+':/release','-v',str(W)+':/quality:ro','-v','/home/valentine/glm53-full-observations-20260907/src:/workspace/src:ro','-v',str(MODEL)+':/candidate:ro']
for x in ['PYTHONPATH=/quality/deps:/workspace/src','GLM53_EXL3_GPU_IDS=0','OMP_NUM_THREADS=2','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','PYTHONUNBUFFERED=1']:cmd+=['-e',x]
cmd += [IMAGE,'/quality/evaluate_mixed_quality.py','--phase','all' if gpu else 'preflight','--artifact','/candidate','--fixture','/release/quality-eval','--teacher','/release/bf16-normalized','--output',RESULT]
cid=subprocess.check_output(cmd,text=True).strip();launch.write_text(json.dumps({'cid':cid,'command':cmd,'time':time.time()},indent=2)+'\n')
(W/(a.phase+'-status.json')).write_text(json.dumps({'state':'RUNNING','cid':cid})+'\n')
if gpu:
 with (W/'monitor.log').open('ab') as f:mon=subprocess.Popen(['python3',str(W/'monitor_capture.py'),'--container',cid,'--result-root',RESULT,'--output',str(W/'layer-history.json')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (W/'monitor-process.json').write_text(json.dumps({'pid':mon.pid,'cid':cid})+'\n')
subprocess.check_output(['docker','wait',cid]);raw=subprocess.check_output(['docker','inspect',cid]);(W/(a.phase+'-inspect.json')).write_bytes(raw)
with (W/(a.phase+'.log')).open('wb') as f:subprocess.run(['docker','logs',cid],stdout=f,stderr=subprocess.STDOUT,check=True)
i=json.loads(raw)[0]
if i['State']['ExitCode']!=0 or i['State']['OOMKilled']:
 (W/(a.phase+'-status.json')).write_text(json.dumps({'state':'FAILED_PRESERVED','cid':cid})+'\n');raise RuntimeError('phase failed; logs retained')
if gpu:
 mon.wait(timeout=30);h=json.loads((W/'layer-history.json').read_text());assert set(map(int,h['layers']))==set(range(45))
 cmd=['docker','run','--name','glm53-projection6-quality-seal','--network','none','--cpus','2','--memory','2g','--memory-swap','2g','--entrypoint','python3','-v',str(ROOT)+':/release:ro','-v',str(W)+':/quality',IMAGE,'/quality/seal_quality_result.py','--result',RESULT,'--inspect','/quality/capture-inspect.json','--history','/quality/layer-history.json','--output','/quality/quality-integrity-seal.json']
 with (W/'seal.log').open('wb') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 for src,dst in [('quality-report.json','quality-report.json'),('evaluation-identity.json','evaluation-identity.json'),('variant-normalized/manifest.json','normalized-manifest.json')]:subprocess.run(['docker','cp',cid+':'+RESULT+'/'+src,str(W/dst)],check=True)
(W/(a.phase+'-status.json')).write_text(json.dumps({'state':'QUALITY_INTEGRITY_SEALED' if gpu else 'PREFLIGHT_PASS','cid':cid})+'\n')
