from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,urllib.request,hashlib,math,collections
import gguf
ROOT=Path(__file__).parent
info=json.loads((ROOT.parent/'unsloth-reference-20260911/gguf-model-info.json').read_text())
files=[f for f in info['siblings'] if f['rfilename'].startswith('UD-IQ3_XXS/')]
class HeaderReader(gguf.GGUFReader):
 def _build_tensors(self,start_offs,fields):
  self.headers=[]
  for f in fields:
   _,name,_,dims,raw_type,offset=f.parts
   typ=gguf.GGMLQuantizationType(raw_type[0]);block,size=gguf.GGML_QUANT_SIZES[typ]
   elements=math.prod(int(d) for d in dims)
   self.headers.append({'name':bytes(name).decode(),'shape':[int(d) for d in dims],'type':typ.name,'elements':elements,'bytes':elements*size//block,'offset':int(start_offs+offset[0])})
def fetch(f):
 size=min(f['size'],10*1024*1024)
 url='https://huggingface.co/'+info['id']+'/resolve/'+info['sha']+'/'+f['rfilename']
 req=urllib.request.Request(url,headers={'Range':f'bytes=0-{size-1}'})
 with urllib.request.urlopen(req,timeout=60) as response:
  data=response.read(size+1);status=response.status;content_range=response.headers.get('Content-Range')
 assert len(data)==size,(len(data),size)
 if size<f['size']:assert status==206 and content_range.startswith(f'bytes 0-{size-1}/'),(status,content_range)
 digest=hashlib.sha256(data).hexdigest()
 if size==f['size']:assert digest==f['lfs']['sha256']
 path=ROOT/(Path(f['rfilename']).name+'.header')
 path.write_bytes(data)
 reader=HeaderReader(path)
 if reader.headers:assert reader.data_offset<=len(data)
 for t in reader.headers:assert t['offset']+t['bytes']<=f['size']
 return {'path':f['rfilename'],'full_size':f['size'],'expected_full_sha256':f['lfs']['sha256'],'header_bytes':len(data),'header_sha256':digest,'full_payload_verified':size==f['size'],'data_offset':int(reader.data_offset),'tensors':reader.headers}
with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,files))
tensors=[t for f in results for t in f['tensors']]
assert len({t['name'] for t in tensors})==len(tensors)
by_type=collections.defaultdict(lambda:{'count':0,'bytes':0,'elements':0})
for t in tensors:
 for k in ('bytes','elements'):by_type[t['type']][k]+=t[k]
 by_type[t['type']]['count']+=1
report={'source':info['id'],'revision':info['sha'],'files':results,'by_type':dict(by_type),'tensor_count':len(tensors),'tensor_bytes':sum(t['bytes'] for t in tensors),'weight_file_bytes':sum(f['full_size'] for f in results),'limits':'Header-only HTTP range audit for weight shards. Tensor content/full shard hashes are NOT verified; no model downloaded or executed.'}
(ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
print('tensor examples',[(t['name'],t['type'],t['bytes']) for t in tensors[:25]])
