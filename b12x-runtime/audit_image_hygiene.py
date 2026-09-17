import json,os,re
from pathlib import Path
paths=['/opt/src','/opt/cargo','/opt/rustup','/root/.cache/uv','/root/.ssh','/root/.aws','/root/.config/gh','/root/.cache/huggingface/token','/vllm-workspace/build','/tmp/exllamav3']
rows={p:{'exists':Path(p).exists()} for p in paths}
for p,row in rows.items():
 if row['exists'] and p in ['/opt/src','/opt/cargo','/opt/rustup','/root/.cache/uv','/vllm-workspace/build','/tmp/exllamav3']:
  files=[f for f in Path(p).rglob('*') if f.is_file() and not f.is_symlink()]
  row.update(files=len(files),bytes=sum(f.stat().st_size for f in files))
pins=json.loads(Path('/opt/source-pins.json').read_text())
print(json.dumps({'state':'SCOPED_FILESYSTEM_AND_ENV_AUDIT_NOT_FULL_PUBLICATION_CLEARANCE','paths':rows,'credential_like_environment_names':[k for k in os.environ if re.search(r'(TOKEN|PASSWORD|SECRET|API_KEY|PRIVATE_KEY)',k,re.I)],'private_observer_pin_present':'observer_base_image_id_2822' in pins,'protected_head_flags':{k:os.environ.get(k) for k in ['VLLM_MXFP8_LM_HEAD','VLLM_MTP_NVFP4_LM_HEAD']}},indent=2))
