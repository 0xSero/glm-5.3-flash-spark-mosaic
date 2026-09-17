import glob,json,struct
from pathlib import Path
import torch
from vllm.model_executor.model_loader.weight_utils import filter_duplicate_safetensors_files
assert not torch.cuda.is_initialized()
root=Path('/model');idx=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
files=filter_duplicate_safetensors_files(glob.glob('/model/*.safetensors'),'/model','model.safetensors.index.json')
assert len(files)==133
headers={}
for name in sorted(set(idx.values())):
 with (root/name).open('rb') as f:headers[name]=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
for key,name in idx.items():assert key in headers[name],key
mi=json.loads(Path('/mtp/model.safetensors.index.json').read_text())['weight_map'];mh={}
for name in set(mi.values()):
 with (Path('/mtp')/name).open('rb') as f:mh[name]=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
for key,name in mi.items():assert key in mh[name],key
assert len(mi)==891 and not torch.cuda.is_initialized()
print(json.dumps({'passed':True,'actual_stock_loader_filter':True,'target_files':len(files),'target_index_tensors':len(idx),'native_draft_tensors':len(mi),'all_headers_and_container_links_resolved':True,'cuda_initialized':False}))
