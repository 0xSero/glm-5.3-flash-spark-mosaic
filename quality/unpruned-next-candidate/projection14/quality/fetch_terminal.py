"""Fetch and verify terminal quality receipts after the remote integrity seal."""
import hashlib,io,json,math,subprocess,tarfile,time
from pathlib import Path
W=Path(__file__).resolve().parent;REMOTE='/home/valentine/glm53-single-spark-release-20260911/quality/unpruned-next-candidate/projection14/quality'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
while True:
 check=subprocess.run(['ssh','-o','ConnectTimeout=10','spark-de5c','cat',REMOTE+'/capture-status.json'],capture_output=True,text=True,timeout=20)
 if check.returncode==0:
  status=json.loads(check.stdout)
  if status['state']=='QUALITY_INTEGRITY_SEALED':break
 # Do not interpret an unavailable status file as successful completion.
 state=subprocess.run(['ssh','-o','ConnectTimeout=10','spark-de5c','cat',REMOTE+'/capture-status.json'],capture_output=True,text=True,timeout=20)
 if state.returncode==0 and json.loads(state.stdout)['state']=='FAILED_PRESERVED':raise RuntimeError('capture failed; remote logs retained')
 time.sleep(30)
files=['quality-report.json','evaluation-identity.json','normalized-manifest.json','quality-integrity-seal.json','capture-status.json','capture-inspect.json','layer-history.json','capture.log','seal.log']
data=subprocess.check_output(['ssh','spark-de5c','tar','-C',REMOTE,'-cf','-',*files]);out=W/'terminal-receipts';out.mkdir(exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(data)) as t:t.extractall(out,filter='data')
r=json.loads((out/'quality-report.json').read_text());s=json.loads((out/'quality-integrity-seal.json').read_text());i=json.loads((out/'capture-inspect.json').read_text())[0];h=json.loads((out/'layer-history.json').read_text())
assert i['State']['ExitCode']==0 and not i['State']['OOMKilled']
assert r['candidate_manifest_sha256']=='80967e71922a534a3ab2872ad90edc4aacaf958bcb0a060699e599318727556d'
assert r['adapter_sha256']==sha(W/'evaluate_mixed_quality.py') and r['mixed_validator_sha256']==sha(W/'mixed_contract.py')
assert r['tokens']==r['positions']==65504 and r['experts_per_layer']==288 and len(r['per_row'])==32
assert s['quality_report_sha256']==sha(out/'quality-report.json') and s['normalized_manifest_sha256']==sha(out/'normalized-manifest.json')
assert s['identity_sha256']==sha(out/'evaluation-identity.json')
assert s['container_id']==i['Id']==h['container_id']
assert s['raw_receipt_coverage_complete'] and sorted(map(int,h['layers']))==list(range(45))
metrics={k:r[k] for k in ['kl_bf16_to_variant','top1_agreement','bf16_perplexity','variant_perplexity']};assert all(math.isfinite(v) for v in metrics.values())
summary={'state':'MATCHED_QUALITY_MEASURED_AND_INTEGRITY_SEALED','candidate_manifest_sha256':r['candidate_manifest_sha256'],'container_id':i['Id'],'positions':65504,**metrics,'quality_accepted':False,'runtime_accepted':False,'receipt_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}}
(W/'FINAL_RESULT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
