"""Preserve bounded CPU build completion and verify its structural seal."""
import hashlib,json,subprocess,time
from pathlib import Path
r=Path('/home/valentine/glm53-single-spark-release-20260911');work=r/'build-q3-k192'
launch=json.loads((work/'launch.json').read_text());cid=launch['container_id']
code=subprocess.check_output(['docker','wait',cid],text=True).strip()
inspect=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
(work/'final-inspect.json').write_text(json.dumps(inspect,indent=2)+'\n')
with (work/'build.log').open('wb') as log:subprocess.run(['docker','logs',cid],stdout=log,stderr=subprocess.STDOUT,check=True)
proof={'state':'FAILED_PRESERVED','exit_code':code,'container_id':cid,'finished_unix':time.time()}
try:
 assert inspect['State']['ExitCode']==0 and not inspect['State']['OOMKilled']
 root=r/'q3-massmax-k192'
 status=json.loads((root/'BUILD_STATUS.json').read_text());manifest=json.loads((root/'EXL3_MANIFEST.json').read_text())
 digest=hashlib.sha256((root/'EXL3_MANIFEST.json').read_bytes()).hexdigest()
 assert status['state']=='COMPLETE' and status['manifest_sha256']==digest
 assert manifest['state']=='STRUCTURAL_PASS' and manifest['keep']==192 and manifest['target_bpw']==3.0
 assert manifest['native_mtp_n_routed_experts']==288 and len(manifest['files'])==130
 config=json.loads((root/'config.json').read_text())
 assert config['text_config']['n_routed_experts']==192 and config['quantization_config']['bits']==3
 for name,sha in manifest['metadata_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
 proof.update(state='STRUCTURAL_COMPLETE_QUALITY_NOT_RUN',manifest_sha256=digest,tensor_bytes=manifest['tensor_bytes'],
              file_bytes=sum(x['bytes'] for x in manifest['files']),files=len(manifest['files']),
              tensor_counts=manifest['tensor_counts'],runtime_claim=False,quality_claim=False)
except Exception as exc:proof['error']=repr(exc)
(work/'completion.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
