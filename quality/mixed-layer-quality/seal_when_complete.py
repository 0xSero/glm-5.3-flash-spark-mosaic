"""Read-only terminal capture validation; leaves all failed artifacts intact."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/valentine/glm53-single-spark-release-20260911');W=ROOT/'quality/mixed-layer-quality';IMAGE='sha256:54acf16064726697ab079dfbd297c1c1c260cb28867f8287f5ae2ec3bbb0a382'
while True:
 status=json.loads((W/'status.json').read_text())
 if status['state']=='FAILED_PRESERVED':raise RuntimeError('quality failed; inspect retained stage logs')
 if status['state']=='CAPTURE_COMPLETE_PENDING_INTEGRITY_SEAL':break
 time.sleep(20)
for _ in range(6):
 history=json.loads((W/'layer-history.json').read_text())
 if not history.get('container_state',{}).get('Running',True):break
 time.sleep(5)
assert set(map(int,history['layers']))==set(range(45)), 'missing raw layer receipts; explicit audit needed'
cmd=['docker','run','--name','glm53-mixed-l5-l32-seal','--network','none','--cpus','2','--memory','2g','--memory-swap','2g','--entrypoint','python3','-v',str(ROOT)+':/release:ro','-v',str(W)+':/quality',IMAGE,'/quality/seal_quality_result.py','--result','/release/results/mixed-k2-k3-l5-l32','--inspect','/quality/capture-inspect.json','--history','/quality/layer-history.json','--output','/quality/quality-integrity-seal.json']
with (W/'seal.log').open('wb') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
# Exited capture container still has the result mounted; copy exact final artifacts
# through Docker to avoid changing ownership of immutable result files.
for src,dst in [('quality-report.json','quality-report.json'),('evaluation-identity.json','evaluation-identity.json'),('variant-normalized/manifest.json','normalized-manifest.json')]:
 subprocess.run(['docker','cp',status['cid']+':/release/results/mixed-k2-k3-l5-l32/'+src,str(W/dst)],check=True)
(W/'seal-status.json').write_text(json.dumps({'state':'INTEGRITY_SEALED_QUALITY_NOT_ACCEPTED','cid':status['cid'],'complete_raw_layer_receipts':45},indent=2)+'\n')
