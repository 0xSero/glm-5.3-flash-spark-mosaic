"""Run on de5c after the observer's safe pause; no model or GPU allocation."""
import json,subprocess
from pathlib import Path
root=Path('/home/valentine/glm53-single-spark-release-20260911')
project=Path('/home/valentine/glm53-full-observations-20260907')
name='glm53-original-records-full-rank0'
assert subprocess.check_output(['docker','wait',name],text=True).strip()=='0'
config=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
cmd=['docker','run','--name','glm53-seal-original-observations','--network','none','--memory','2g','--memory-swap','2g','--cpus','2','-e','PYTHONPATH=/workspace/src','-v',str(project)+':/work:ro','-v',str(project/'src')+':/workspace/src:ro','-v',str(root)+':/release','--entrypoint','python3',config['Image'],'/release/seal_glm_observations.py','--token-manifest','/work/tokens/ours-original-records-v1/manifest.json','--model-identity','/work/q4-original-records.model.json','--run-root','/work/observations-q4-original-records-v1/records-8d8dd5ec0a6608f97d88b158','--output','/release/sealed-observations']
(root/'seal-launch.json').write_text(json.dumps({'argv':cmd,'observer_container':config['Id'],'image':config['Image']},indent=2))
subprocess.run(cmd,check=True)
