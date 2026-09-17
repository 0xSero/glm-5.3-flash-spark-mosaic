"""Fetch the auxiliary serving closure from exact pinned Q3 Hub metadata."""
import hashlib,json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent
REPO='0xSero/GLM-5.3-Flash-EXL3-3.0bpw'
REV='2a30ad09c15f779a44fa62c216f5dbe5fb0c9223'
meta=json.loads((ROOT/'HUB_INVENTORY.json').read_text());assert meta['sha']==REV
files={f['rfilename']:f for f in meta['siblings']}
# Actual pinned file listing: these are the complete tokenizer/processor files.
runtime={n for n in files if ('/' not in n and (n.startswith(('tokenizer','processor','preprocessor','special_tokens','added_tokens','vocab.','merges.','generation_config','chat_template')))) or n.startswith('chat_templates/')}
assert runtime=={'tokenizer.json','tokenizer_config.json','processor_config.json','generation_config.json','chat_template.jinja'}
selected={n:'payload/'+n for n in runtime}
selected.update({n:'provenance/'+n for n in ('LICENSE','THIRD_PARTY_NOTICES.md','PROVENANCE.md')})
selected['config.json']='reference/source-config.json'
records=[]
for name,destination in sorted(selected.items()):
 item=files[name]
 data=urllib.request.urlopen(f'https://huggingface.co/{REPO}/resolve/{REV}/{name}',timeout=60).read()
 assert len(data)==item['size'],name
 digest=hashlib.sha256(data).hexdigest()
 if item.get('lfs'):
  assert digest==item['lfs']['sha256'],name
 else:
  assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==item['blobId'],name
 path=ROOT/destination;path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.read_bytes()==data
 else:path.write_bytes(data)
 records.append({'path':destination,'source_file':name,'bytes':len(data),'sha256':digest,'hub_blob_id':item['blobId'],'hub_lfs_sha256':item.get('lfs',{}).get('sha256')})
config=json.loads((ROOT/'reference/source-config.json').read_text())
tokenizer=json.loads((ROOT/'payload/tokenizer_config.json').read_text())
processor=json.loads((ROOT/'payload/processor_config.json').read_text())
assert not config.get('auto_map') and not tokenizer.get('auto_map') and not processor.get('auto_map')
manifest={'schema':'glm53-pinned-q3-runtime-addon-v1','state':'HUB_HASH_VERIFIED','repo':REPO,'revision':REV,
 'hub_inventory_sha256':hashlib.sha256((ROOT/'HUB_INVENTORY.json').read_bytes()).hexdigest(),
 'files':records,'runtime_payload_files':sorted(runtime),'trusted_remote_code_required':False,
 'tokenizer_class':tokenizer.get('tokenizer_class'),'processor_class':processor.get('processor_class'),
 'core_metadata_policy':'No sealed model config/index/manifest changes. payload/ is the auxiliary serving addon; reference/ and provenance/ are separate evidence.',
 'other_hub_files':'Complete Hub listing retained; weights, old layer sidecars, calibration/evaluation evidence and standalone old runtime helper are not tokenizer/processor dependencies.'}
(ROOT/'ADDON_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
