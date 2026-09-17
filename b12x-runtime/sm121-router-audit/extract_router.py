import hashlib,json,pathlib
from safetensors import safe_open
from safetensors.torch import save_file
source=pathlib.Path('/model/k2/retained-00109-of-00120.safetensors')
prefix='model.language_model.layers.5.mlp.gate.'
with safe_open(source,framework='pt',device='cpu') as f:
 tensors={'weight':f.get_tensor(prefix+'weight'),'bias':f.get_tensor(prefix+'e_score_correction_bias')}
assert tuple(tensors['weight'].shape)==(288,4096)
out=pathlib.Path('/out');out.mkdir(exist_ok=True)
save_file(tensors,str(out/'router.safetensors'))
receipt={'source_relative_path':'k2/retained-00109-of-00120.safetensors','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'tensor_keys':[prefix+'weight',prefix+'e_score_correction_bias'],'tensors':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in tensors.items()},'router_sha256':hashlib.sha256((out/'router.safetensors').read_bytes()).hexdigest()}
(out/'router-source.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
