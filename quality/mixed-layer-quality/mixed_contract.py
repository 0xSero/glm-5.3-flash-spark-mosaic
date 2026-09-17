"""Fail-closed validation of the exact assembled mixed5+32 candidate."""
import hashlib,json,re,struct
from pathlib import Path
MANIFEST_SHA='73291ede5c0f2ad13dcd9fbdfe3887f133ef713d56c589c59c6392bdb887b45c'
K2_SHA='501641b947fa56afc9ed098bdc034e59c2bd229d4fa1a1d7b80e3727f10c8ba3'
K3_SHA='4a4402ab7b47f91145db841ce0105b54c3f7fcc3663a110fa712ceb85d5d1f0f'
PREFIX=re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.rank([0-3])$')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def require(value,message):
 if not value:raise ValueError(message)
def expected_bits(prefix):
 m=PREFIX.fullmatch(prefix)
 require(m is not None,'not a routed packed expert prefix')
 layer,expert=int(m[1]),int(m[2])
 require(3<=layer<=44 and 0<=expert<288,'routed expert out of range')
 return 3 if layer in (5,32) else 2
def validate_trellis_shape(prefix,shape):
 require(len(shape)==3 and shape[-1]==16*expected_bits(prefix),'trellis bits differ from layer map')
def checked(root,name):
 p=Path(name)
 require(not p.is_absolute() and '..' not in p.parts,'unsafe manifest path')
 p=root/p;require(p.is_file(),'missing manifest file');return p
def validate_candidate(root,experts):
 root=Path(root);mp=root/'EXL3_MANIFEST.json'
 require(sha(mp)==MANIFEST_SHA,'not sealed mixed5+32 manifest')
 m=json.loads(mp.read_text());s=json.loads((root/'BUILD_STATUS.json').read_text())
 require(experts==288 and m['schema']=='glm53-mixed-whole-layer-exl3-v1' and m['state']=='STRUCTURAL_PASS','mixed source contract mismatch')
 require(m['source_k2_manifest_sha256']==K2_SHA and m['paired_k3_encoded_status_sha256']==K3_SHA,'mixed source identity mismatch')
 require(s['state']=='COMPLETE' and s['manifest_sha256']==MANIFEST_SHA,'missing complete build seal')
 require(m['layer_bits']=={'5':3,'32':3} and m['default_bits']==2,'incorrect precision map')
 require(m['native_mtp_experts']==288 and m['protected_tensor_bytes']==33835039608,'native protected closure mismatch')
 for n,digest in m['metadata_sha256'].items():require(sha(checked(root,n))==digest,'metadata digest mismatch: '+n)
 c=json.loads((root/'config.json').read_text());q=json.loads((root/'quantization_config.json').read_text())
 require(c['quantization_config']==q and q['bits']==2 and q['layer_bits']==m['layer_bits'] and q['rank_stacked_tp']==4,'runtime precision metadata differs')
 index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];actual={};total=protected=0
 for f in m['files']:
  p=checked(root,f['path']);require(p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],'payload digest mismatch: '+f['path'])
  with p.open('rb') as h:
   length=struct.unpack('<Q',h.read(8))[0];require(2<=length<=64<<20,'invalid header size');header=json.loads(h.read(length))
  for name,v in header.items():
   if name=='__metadata__':continue
   require(name not in actual and index.get(name)==f['path'],'tensor index disagrees')
   actual[name]=f['path'];size=v['data_offsets'][1]-v['data_offsets'][0];total+=size
   prefix,suffix=name.rsplit('.',1)
   if PREFIX.fullmatch(prefix):
    bits=expected_bits(prefix);require(f['source_tier']==('k3' if bits==3 else 'k2'),'wrong source tier')
    if suffix=='trellis':validate_trellis_shape(prefix,v['shape']);require(v['dtype']=='I16','trellis dtype changed')
   else:protected+=size
 require(actual==index and len(actual)==583090,'incomplete tensor coverage')
 require(total==113086732152 and protected==33835039608,'tensor payload closure mismatch')
 return MANIFEST_SHA
