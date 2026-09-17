"""Container-compatible loader view: identical hardlinked tensors, remapped index only."""
import hashlib,json,os,shutil
from pathlib import Path
root=Path('/home/valentine/glm53-single-spark-release-20260911')
src=root/'quality/unpruned-next-candidate/projection14/model';out=root/'b12x-runtime/mtp-nvfp4-serving/flat-projection14'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(src/'EXL3_MANIFEST.json')=='80967e71922a534a3ab2872ad90edc4aacaf958bcb0a060699e599318727556d'
out.mkdir(exist_ok=False)
manifest=json.loads((src/'EXL3_MANIFEST.json').read_text());index=json.loads((src/'model.safetensors.index.json').read_text());mapping=index['weight_map'];names={Path(n).name for n in mapping.values()};assert len(names)==len(set(mapping.values()))==133
for p in src.iterdir():
 if p.is_file() and p.name!='model.safetensors.index.json':shutil.copy2(p,out/p.name)
files=[]
for f in manifest['files']:
 p=src/f['path'];target=out/p.name
 assert p.stat().st_size==f['bytes']
 os.link(p,target)
 assert os.path.samefile(p,target)
 files.append({'source':str(p),'target':str(target),'inode':p.stat().st_ino,'bytes':p.stat().st_size,'source_manifest_sha256':f['sha256']})
# Keep original native MTP /model/weights/... links and manifest paths valid.
(out/'weights').symlink_to('.',target_is_directory=True)
index['weight_map']={k:Path(v).name for k,v in mapping.items()}
(out/'model.safetensors.index.json').write_text(json.dumps(index,indent=2)+'\n')
for k,v in mapping.items():assert os.path.samefile(src/v,out/index['weight_map'][k])
(out/'SERVING_VIEW.json').write_text(json.dumps({'state':'IDENTICAL_TENSOR_INODES_LOADER_VIEW','source':str(src),'source_manifest_sha256':sha(src/'EXL3_MANIFEST.json'),'source_index_sha256':sha(src/'model.safetensors.index.json'),'view_index_sha256':sha(out/'model.safetensors.index.json'),'tensor_values_modified':False,'source_index_modified':False,'files':files},indent=2)+'\n')
print(json.dumps({'view':str(out),'files':len(files),'view_index_sha256':sha(out/'model.safetensors.index.json')}))
