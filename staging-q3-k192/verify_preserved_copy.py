"""Read-only full backup proof before deleting a specifically authorized temp copy."""
import hashlib,json,time
from pathlib import Path
r=Path('/home/valentine/glm53-single-spark-release-20260911/q3-massmax-k176')
PIN='0230ee28b89f0eab7b24bebca8ddd10ff3e96783f408efe17ca1a9bbe0f39a13'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
assert sha(r/'EXL3_MANIFEST.json')==PIN
manifest=json.loads((r/'EXL3_MANIFEST.json').read_text());status=json.loads((r/'BUILD_STATUS.json').read_text())
assert status['state']=='COMPLETE' and status['manifest_sha256']==PIN
assert manifest['keep']==176 and manifest['state']=='STRUCTURAL_PASS'
for name,expected in manifest['metadata_sha256'].items():assert sha(r/name)==expected,name
verified=[]
for item in manifest['files']:
 if item.get('omitted_empty_shard'):continue
 path=r/item['path'];assert not path.is_symlink() and r in path.resolve().parents
 assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path']
 verified.append({'path':item['path'],'bytes':item['bytes'],'sha256':item['sha256']})
print(json.dumps({'state':'PRESERVED_COPY_FULL_HASH_PASS','manifest_sha256':PIN,'source':str(r),
                 'files':verified,'total_bytes':sum(x['bytes'] for x in verified),'verified_unix':time.time()}))
