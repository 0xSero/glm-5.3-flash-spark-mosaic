#!/usr/bin/env python3
"""Immutable, CPU-only layer5/32 K3 replacement in the pinned complete K2 model."""
import argparse,errno,hashlib,json,os,re,shutil,struct,time
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
K2_MANIFEST='501641b947fa56afc9ed098bdc034e59c2bd229d4fa1a1d7b80e3727f10c8ba3'
K2_INVENTORY='36b79ee52ace17008eb917588666e3ae7f6ce5f419470df019da8a27576d594b'
K3_STATUS='4a4402ab7b47f91145db841ce0105b54c3f7fcc3663a110fa712ceb85d5d1f0f'
INPUT_SHA256={'paired-k3-availability.json': '41ae0821e39a7a20343b73ec9f45c2c473330f874465326e56bccd5ff0daae8f', 'paired-proxy-inventory.json': '78e602f9ee18d25a4a5334d0ba09564f2a90d6601e81fb4e65be9972e5b16dd3', 'source-metadata/auxiliary-manifest.json': 'd22f612e9e0f0c7aa6a25394ee1395319cc401420048c11b75dc3d3faa46ca19'}
LAYERS={5,32};PROTECTED=33835039608;TOTAL=113086732152
EXPERT=re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.rank([0-3])\.(trellis|suh|svh|mcg)$')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(p)
def header(p):
 with p.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0]
  if not 2<=n<=min(p.stat().st_size-8,64<<20):raise ValueError('invalid header size')
  h=json.loads(f.read(n))
 cursor=0
 for k,v in sorted(((k,v) for k,v in h.items() if k!='__metadata__'),key=lambda kv:kv[1]['data_offsets']):
  lo,hi=v['data_offsets']
  if lo!=cursor or hi<lo:raise ValueError('invalid tensor offsets')
  cursor=hi
 if cursor+n+8!=p.stat().st_size:raise ValueError('payload size mismatch')
 return {k:v for k,v in h.items() if k!='__metadata__'}
def verified_materialize(src,dst,expected):
 src=src.resolve(strict=True)
 if src.stat().st_size!=expected['bytes'] or sha(src)!=expected['sha256']:raise ValueError('source digest mismatch: '+str(src))
 if dst.exists():raise ValueError('refuse overwrite: '+str(dst))
 dst.parent.mkdir(parents=True,exist_ok=True);temp=dst.with_name(dst.name+'.incoming')
 if temp.exists():raise ValueError('partial output already exists')
 try:os.link(src,temp);mode='hardlink'
 except OSError as e:
  if e.errno!=errno.EXDEV:raise
  with src.open('rb') as f,temp.open('xb') as g:shutil.copyfileobj(f,g,8<<20)
  mode='copy'
 if sha(temp)!=expected['sha256']:raise ValueError('destination digest mismatch')
 temp.replace(dst);return mode

def config_with_layers(config,quant):
 q=dict(quant);q.update(bits=2,layer_bits={'5':3,'32':3},rank_stacked_tp=4,effective_routed_tier_bpw=2+2/42,target_routed_tier_bpw=2+2/42,)
 q.pop('k2_experts_per_layer',None);q.pop('k3_experts_per_layer',None)
 q['selection']={'type':'whole_layer_from_independent_calibration_proxy','upgraded_layers':[5,32]}
 q['precision_layout']='uniform_within_each_routed_layer';q['k2_layer_count']=40;q['k3_layer_count']=2
 c=dict(config);c['quantization_config']=q;c['native_mtp_n_routed_experts']=288
 return c,q

def build(a):
 for name,expected in INPUT_SHA256.items():assert sha(HERE/name)==expected,('input metadata changed',name)
 meta=HERE/'source-metadata';assert sha(meta/'EXL3_MANIFEST.json')==K2_MANIFEST and sha(meta/'filehash-inventory.json')==K2_INVENTORY
 inv=read(meta/'filehash-inventory.json');original=read(meta/'EXL3_MANIFEST.json');index=read(meta/'model.safetensors.index.json')['weight_map']
 for n,expected in inv['metadata_sha256'].items():
  if n in ['EXL3_MANIFEST.json','config.json','quantization_config.json','model.safetensors.index.json']:assert sha(meta/n)==expected
 availability=read(HERE/'paired-k3-availability.json');assert availability['status_sha256']==K3_STATUS
 assert availability['status']['session_calibration_tokens_sha256']==original['calibration']['token_rows_sha256']
 mapping=read(a.source_map);out=a.output.resolve();assert not out.exists();sources=[Path(p).resolve() for p in mapping.values()]
 assert all(out!=p and out not in p.parents for p in sources)
 aux=read(meta/'auxiliary-manifest.json')
 for n,r in aux['files'].items():assert sha(meta/n)==r['sha256'] and (meta/n).stat().st_size==r['bytes']
 files=[];sourcespec=[]
 for rec in inv['files']:
  match=re.match(r'layer-(\d+)-part-tail-\d+.safetensors$',rec['path'])
  if match and int(match[1]) in LAYERS:continue
  sourcespec.append({'path':'k2/'+rec['path'],'source_key':rec['path'],'bits':2,'kind':'k2',**{k:rec[k] for k in ['bytes','sha256']}})
 for rec in availability['parts']:
  if rec['layer'] in LAYERS:
   sourcespec.append({'path':f"k3/layer-{rec['layer']:02d}-part-{rec['gpu']}.safetensors",'layer':rec['layer'],'bits':3,'kind':'k3','bytes':rec['declared_bytes'],'sha256':rec['declared_sha256']})
 assert len(sourcespec)==133
 # Admission covers worst-case copies, including source mappings on other filesystems.
 need=sum(x['bytes'] for x in sourcespec if Path(mapping[x['sha256']]).stat().st_dev!=out.parent.stat().st_dev)
 assert shutil.disk_usage(out.parent).free>=need+20*2**30
 if a.plan_only:
  print(json.dumps({'state':'PLAN_ONLY_NO_OUTPUT_CREATED','files':len(sourcespec),'weight_file_bytes':sum(x['bytes'] for x in sourcespec),'expected_tensor_bytes':TOTAL,'cross_filesystem_copy_bytes':need,'output':str(out),'upgraded_layers':[5,32]}));return
 out.mkdir();write(out/'BUILD_STATUS.json',{'state':'BUILDING','files':0});weight_map={};counts=Counter();protected_names=[];tensor_bytes=0;protected_bytes=0
 for number,rec in enumerate(sourcespec,1):
  src=Path(mapping[rec['sha256']]);dst=out/rec['path'];mode=verified_materialize(src,dst,rec);h=header(dst);filebytes=0
  for name,value in h.items():
   if name in weight_map:raise ValueError('duplicate tensor: '+name)
   m=EXPERT.match(name)
   if rec['kind']=='k3':assert m and int(m[1])==rec['layer'] and 0<=int(m[2])<288
   else:assert index.get(name)==rec['source_key']
   if m:
    layer,expert=int(m[1]),int(m[2]);assert 3<=layer<=44 and 0<=expert<288
    bits=3 if layer in LAYERS else 2;assert bits==rec['bits']
    if m[5]=='trellis':assert value['dtype']=='I16' and value['shape'][-1]==16*bits
    elif m[5] in ['suh','svh']:assert value['dtype']=='F16'
    else:assert value['dtype']=='I32'
    counts[(layer,expert)]+=1
   else:
    assert rec['kind']=='k2';protected_names.append(name);protected_bytes+=value['data_offsets'][1]-value['data_offsets'][0]
   size=value['data_offsets'][1]-value['data_offsets'][0];filebytes+=size;weight_map[name]=rec['path']
  tensor_bytes+=filebytes;files.append({k:rec[k] for k in ['path','bytes','sha256']}|{'tensor_bytes':filebytes,'tensor_count':len(h),'materialization':mode,'source_tier':rec['kind']})
  write(out/'BUILD_STATUS.json',{'state':'BUILDING','files':number,'tensor_bytes':tensor_bytes});print(json.dumps({'file':number,'of':133,'mode':mode}),flush=True)
 assert set(weight_map)==set(index) and len(weight_map)==583090
 assert len(counts)==42*288 and set(counts.values())=={48}
 assert protected_bytes==PROTECTED and tensor_bytes==TOTAL and len(protected_names)==2482
 # Original layer45 experts are native names and remain entirely within protected closure.
 assert len([n for n in protected_names if n.startswith('model.language_model.layers.45.')])==889
 c,q=config_with_layers(read(meta/'config.json'),read(meta/'quantization_config.json'))
 for n in aux['files']:shutil.copyfile(meta/n,out/n)
 shutil.copyfile(meta/'config.json',out/'source-config.json');write(out/'config.json',c);write(out/'quantization_config.json',q)
 write(out/'model.safetensors.index.json',{'metadata':{'total_size':tensor_bytes},'weight_map':weight_map})
 write(out/'protected-tensor-closure.json',{'state':'SOURCE_BYTE_EXACT_FILE_HASH_VERIFIED','tensor_bytes':protected_bytes,'tensor_count':2482,'native_mtp_tensor_count':889,'tensor_names':sorted(protected_names)})
 metadata={n:sha(out/n) for n in ['config.json','source-config.json','quantization_config.json','model.safetensors.index.json','protected-tensor-closure.json',*aux['files']]}
 manifest={'schema':'glm53-mixed-whole-layer-exl3-v1','state':'STRUCTURAL_PASS','source_k2_manifest_sha256':K2_MANIFEST,'paired_k3_encoded_status_sha256':K3_STATUS,'layer_bits':q['layer_bits'],'default_bits':2,'experts_per_layer':288,'native_mtp_experts':288,'tensor_bytes':tensor_bytes,'weight_file_bytes':sum(x['bytes'] for x in files),'protected_tensor_bytes':protected_bytes,'files':files,'metadata_sha256':metadata,'calibration_manifest_sha256':original['calibration']['manifest_sha256'],'selection_policy':'independent600-row calibration sum K2-minus-K3 proxy ranking, layers5+32; not selected on heldout WikiText','selection_proxy_inventory_sha256':sha(HERE/'paired-proxy-inventory.json'),'builder_sha256':sha(__file__),'quality_accepted':False,'runtime_accepted':False}
 write(out/'EXL3_MANIFEST.json',manifest);write(out/'BUILD_STATUS.json',{'state':'COMPLETE','files':133,'tensor_bytes':tensor_bytes,'manifest_sha256':sha(out/'EXL3_MANIFEST.json')});print(json.dumps(read(out/'BUILD_STATUS.json')),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-map',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args();build(a)
