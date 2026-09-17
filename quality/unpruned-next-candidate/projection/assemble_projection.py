#!/usr/bin/env python3
"""Immutable CPU projection6 assembly; all target expert IDs/router rows retained."""
import hashlib,importlib.util,json,os,re,shutil,struct
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OLD=ROOT/'quality/mixed-layer-candidate';META=OLD/'source-metadata'
SELECTION_SHA='38efe0f4da5d382499d4a3fb220ac66c8597b59190e11cd7679038f21afc6567'
if hashlib.sha256((OLD/'assemble.py').read_bytes()).hexdigest()!='c3e0956334da7c803b98ac6845d741fbeddba4548a4b97ed366b97ea743416ca':raise ValueError('helper source changed')
spec=importlib.util.spec_from_file_location('projection_helpers',OLD/'assemble.py');H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
EXPERT=re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.rank([0-3])\.(trellis|suh|svh|mcg)$')
def rewrite(src2,src3,dst,rec2,rec3):
 for p,r in ((src2,rec2),(src3,rec3)):
  if p.stat().st_size!=r['bytes'] or H.sha(p)!=r['sha256']:raise ValueError('source hash mismatch')
 h2=H.header(src2);h3=H.header(src3)
 if set(h2)!=set(h3):raise ValueError('paired source tensor names differ')
 headers={};spans=[];cursor=0;changed=0
 for name,v2 in sorted(h2.items(),key=lambda kv:kv[1]['data_offsets']):
  m=EXPERT.fullmatch(name)
  if not m:raise ValueError('selected part contains non-expert tensor')
  down=m[3]=='down_proj';v=h3[name] if down else v2
  if v['dtype']!=v2['dtype']:raise ValueError('paired source dtype differs')
  if down and m[5]=='trellis':
   if v['shape'][:-1]!=v2['shape'][:-1] or v['shape'][-1]!=48 or v2['shape'][-1]!=32:raise ValueError('bad down projection precision')
  elif v['shape']!=v2['shape']:raise ValueError('unexpected paired shape difference')
  size=v['data_offsets'][1]-v['data_offsets'][0]
  headers[name]={'dtype':v['dtype'],'shape':v['shape'],'data_offsets':[cursor,cursor+size]};spans.append((down,v['data_offsets'][0],size));cursor+=size;changed+=int(down)
 header=json.dumps(headers,separators=(',',':')).encode();header+=b' '*((-len(header))%8)
 dst.parent.mkdir(parents=True,exist_ok=True);temp=dst.with_suffix('.incoming')
 if dst.exists() or temp.exists():raise ValueError('refuse overwrite')
 with src2.open('rb') as f2,src3.open('rb') as f3,temp.open('xb') as out:
  off2=8+struct.unpack('<Q',f2.read(8))[0];off3=8+struct.unpack('<Q',f3.read(8))[0];out.write(struct.pack('<Q',len(header)));out.write(header)
  for down,offset,size in spans:
   f=f3 if down else f2;f.seek((off3 if down else off2)+offset)
   while size:
    b=f.read(min(size,8<<20))
    if not b:raise ValueError('short source read')
    out.write(b);size-=len(b)
 temp.replace(dst)
 if H.header(dst)!=headers:raise ValueError('rewritten header mismatch')
 return changed

def main():
 if H.sha(HERE/'selection.json')!=SELECTION_SHA:raise ValueError('selection changed')
 sel=H.read(HERE/'selection.json');selected=set(map(int,sel['layer_projection_bits']));assert selected=={5,26,31,32,35,36}
 if H.sha(META/'EXL3_MANIFEST.json')!=H.K2_MANIFEST or H.sha(META/'filehash-inventory.json')!=H.K2_INVENTORY:raise ValueError('source metadata changed')
 inv=H.read(META/'filehash-inventory.json')
 for n in ('config.json','quantization_config.json','model.safetensors.index.json','EXL3_MANIFEST.json'):
  if H.sha(META/n)!=inv['metadata_sha256'][n]:raise ValueError('source metadata changed: '+n)
 mapping=H.read(OLD/'source-map.json');index=H.read(META/'model.safetensors.index.json')['weight_map'];parts={(x['layer'],x['part']):x for x in sel['selected_source_parts']}
 out=HERE/'model';assert not out.exists()
 required=0
 for rec in inv['files']:
  m=re.fullmatch(r'layer-(\d+)-part-tail-(\d+)\.safetensors',rec['path'])
  if m and int(m[1]) in selected:required+=rec['bytes']+144*1048576
  elif Path(mapping[rec['sha256']]).stat().st_dev!=HERE.stat().st_dev:required+=rec['bytes']
 if shutil.disk_usage(HERE).free<required+20*2**30:raise ValueError('insufficient build/disk floor')
 out.mkdir();files=[];weights={};protected=[];total=protected_bytes=0;replaced_fields=0
 for i,rec in enumerate(inv['files'],1):
  relative='weights/'+rec['path'];dst=out/relative;src2=Path(mapping[rec['sha256']]);m=re.fullmatch(r'layer-(\d+)-part-tail-(\d+)\.safetensors',rec['path']);contributors=[{'tier':2,'sha256':rec['sha256']}]
  if m and int(m[1]) in selected:
   layer,part=int(m[1]),int(m[2]);source_root=OLD/'k3-source' if layer in (5,32) else HERE/'k3-source';src3=source_root/f'layer-{layer:02d}-part-{part}.safetensors';rec3=parts[(layer,part)]['k3'];assert parts[(layer,part)]['k2']['sha256']==rec['sha256']
   replaced_fields+=rewrite(src2,src3,dst,rec,rec3);mode='projection_rewrite';contributors.append({'tier':3,'sha256':rec3['sha256'],'projection':'down_proj'})
  else:mode=H.verified_materialize(src2,dst,rec)
  header=H.header(dst);fbytes=0
  for name,v in header.items():
   if name in weights or index.get(name)!=rec['path']:raise ValueError('index coverage mismatch')
   size=v['data_offsets'][1]-v['data_offsets'][0];fbytes+=size;weights[name]=relative;expert=EXPERT.fullmatch(name)
   if expert:
    k=3 if int(expert[1]) in selected and expert[3]=='down_proj' else 2
    if expert[5]=='trellis' and v['shape'][-1]!=16*k:raise ValueError('projection bit mismatch')
   else:protected.append(name);protected_bytes+=size
  total+=fbytes;files.append({'path':relative,'bytes':dst.stat().st_size,'sha256':H.sha(dst),'tensor_bytes':fbytes,'tensor_count':len(header),'materialization':mode,'sources':contributors});H.write(out/'BUILD_STATUS.json',{'state':'BUILDING','files':i});print(json.dumps({'file':i,'of':133,'mode':mode}),flush=True)
 assert set(weights)==set(index) and len(weights)==583090 and total==113086732152
 assert len(protected)==2482 and protected_bytes==33835039608 and sum('.layers.45.' in n for n in protected)==889
 assert replaced_fields==6*288*4*4
 c=H.read(META/'config.json');q=H.read(META/'quantization_config.json');q.pop('k2_experts_per_layer',None);q.pop('k3_experts_per_layer',None);q.update(bits=2,rank_stacked_tp=4,layer_projection_bits=sel['layer_projection_bits'],precision_layout='layer_projection_uniform',effective_routed_tier_bpw=2+2/42,target_routed_tier_bpw=2+2/42);c['quantization_config']=q;c['native_mtp_n_routed_experts']=288
 aux=H.read(META/'auxiliary-manifest.json')
 for n,r in aux['files'].items():
  if H.sha(META/n)!=r['sha256']:raise ValueError('auxiliary changed')
  shutil.copyfile(META/n,out/n)
 shutil.copyfile(META/'config.json',out/'source-config.json');H.write(out/'config.json',c);H.write(out/'quantization_config.json',q);H.write(out/'model.safetensors.index.json',{'metadata':{'total_size':total},'weight_map':weights});H.write(out/'protected-tensor-closure.json',{'state':'NATIVE_PROTECTED_BYTE_IDENTICAL','tensor_bytes':protected_bytes,'tensor_count':2482,'native_mtp_tensor_count':889,'tensor_names':sorted(protected)})
 metadata={n:H.sha(out/n) for n in ['config.json','quantization_config.json','source-config.json','model.safetensors.index.json','protected-tensor-closure.json',*aux['files']]}
 manifest={'schema':'glm53-mixed-projection-exl3-v1','state':'STRUCTURAL_PASS','source_k2_manifest_sha256':H.K2_MANIFEST,'paired_k3_encoded_status_sha256':H.K3_STATUS,'selection_sha256':SELECTION_SHA,'layer_projection_bits':sel['layer_projection_bits'],'default_bits':2,'experts_per_layer':288,'native_mtp_experts':288,'protected_tensor_bytes':protected_bytes,'tensor_bytes':total,'weight_file_bytes':sum(f['bytes'] for f in files),'files':files,'metadata_sha256':metadata,'replaced_down_projection_fields':replaced_fields,'builder_sha256':H.sha(Path(__file__)),'helper_sha256':H.sha(Path(H.__file__)),'quality_accepted':False,'runtime_accepted':False}
 H.write(out/'EXL3_MANIFEST.json',manifest);H.write(out/'BUILD_STATUS.json',{'state':'COMPLETE','files':133,'tensor_bytes':total,'manifest_sha256':H.sha(out/'EXL3_MANIFEST.json')});print(json.dumps(H.read(out/'BUILD_STATUS.json')),flush=True)
if __name__=='__main__':main()
