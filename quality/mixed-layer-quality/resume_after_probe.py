"""Resume after corrected probe; preserve the failed first-probe evidence."""
import json,subprocess,shutil
from run_quality import W,run,state
proof=json.loads((W/'reconstruction-proof-r2.json').read_text());assert proof['state']=='REAL_EXL3_RECONSTRUCTION_PASS' and len(proof['projections'])==48
assert {p['inferred_K'] for p in proof['projections']}=={2,3}
raw=subprocess.check_output(['docker','inspect','glm53-mixed-l5-l32-reconstruct-r2']);info=json.loads(raw)[0];assert info['State']['ExitCode']==0 and not info['State']['OOMKilled']
(W/'reconstruct-r2-inspect.json').write_bytes(raw)
with (W/'reconstruct-r2.log').open('wb') as f:subprocess.run(['docker','logs',info['Id']],stdout=f,stderr=subprocess.STDOUT,check=True)
shutil.copyfile(W/'status.json',W/'failed-first-probe-status.json')
common=['--artifact','/candidate','--fixture','/release/quality-eval','--teacher','/release/bf16-normalized','--output','/release/results/mixed-k2-k3-l5-l32']
try:
 cid=run('capture',True,'evaluate_mixed_quality.py',['--phase','all',*common]);state('CAPTURE_COMPLETE_PENDING_INTEGRITY_SEAL',cid=cid)
except Exception as e:state('FAILED_PRESERVED',error=repr(e));raise
