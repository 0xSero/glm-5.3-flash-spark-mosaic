#!/usr/bin/env python3
"""Private frozen BF16 serving reference; never uploads or calls serving APIs."""
import argparse,hashlib,json,math,subprocess,time
from pathlib import Path
FIXTURE='46255621cd2da8e05546edc3498cfbee518da092af2f44ae243f79da97c0f00b'
TOKENS='5b77e320eaccc959e5f731639aa4d9b908027e7647192305e374c945b44c7d44'
TEACHER='a1604015f76d6a6b6f032ffb7e3359a770e8dd2564ca89e3b00eb8bcb4d93314'
ARTIFACT='7c7a2e47c7cde8996ac54c6f2ca5c3f7532b682d5d7296408c97a14e8950f73c'
HEAD='5155051210040e80b6f9a011d7f3e9f68b466fdb1a43af044612b40c34882f5a'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,allow_nan=False,separators=(',',':'));f.write('\n')
def update_best(logits,start,best,arg):
 maximum,indices=logits.max(dim=-1)
 take=maximum>best  # Strict > preserves lower token ID on inter-chunk ties.
 arg[take]=indices[take]+start;best[take]=maximum[take]
def self_test():
 import torch
 best=torch.full((2,),-float('inf'));arg=torch.zeros((2,),dtype=torch.long)
 update_best(torch.tensor([[1.,1.],[0.,2.]]),0,best,arg)
 update_best(torch.tensor([[1.,0.],[2.,3.]]),2,best,arg)
 assert arg.tolist()==[0,3]
 assert not torch.cuda.is_initialized()
 print('CPU_TIE_STABLE_ARGMAX_PASS')
def prepare(a):
 root=a.release;out=a.output;fixture=root/'quality-eval';norm=root/'bf16-normalized';candidate=root/'k2-massmax-k256'
 assert not (out/'identity.json').exists()
 assert sha(fixture/'manifest.json')==FIXTURE and sha(fixture/'token_rows.safetensors')==TOKENS
 assert sha(norm/'manifest.json')==TEACHER and sha(candidate/'EXL3_MANIFEST.json')==ARTIFACT
 manifest=read(norm/'manifest.json');assert manifest['state']=='COMPLETE' and manifest['rows']==32
 assert [r['row'] for r in manifest['records']]==list(range(32))
 for r in manifest['records']:
  p=norm/f"row-{r['row']:03d}.safetensors";assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
 index=read(candidate/'model.safetensors.index.json')['weight_map'];name=index['lm_head.weight']
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 headpath=candidate/name;assert sha(headpath)==HEAD
 declared=next(x for x in read(candidate/'EXL3_MANIFEST.json')['files'] if x['path']==name)
 assert declared['sha256']==declared['source_sha256']==HEAD
 baseline=read(root/'staging-k2-k256/original-q3-results-seal.json')['scores']
 assert baseline['teacher_manifest_sha256']==TEACHER and baseline['fixture_manifest_sha256']==FIXTURE
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 import torch
 import torch.nn.functional as F
 from safetensors import safe_open
 from safetensors.torch import load_file
 tokens=load_file(fixture/'token_rows.safetensors')['input_ids'];assert tokens.shape==(32,2048) and tokens.dtype==torch.int64
 with safe_open(str(headpath),framework='pt',device='cpu') as f:head=f.get_tensor('lm_head.weight')
 assert head.dtype==torch.bfloat16 and head.shape[1]==4096 and head.ndim==2
 assert tokens.min()>=0 and tokens.max()<head.shape[0]
 top1=[];nll=[];started=time.time()
 with torch.inference_mode():
  for row in range(32):
   hidden=load_file(norm/f'row-{row:03d}.safetensors')['hidden']
   assert hidden.dtype==torch.bfloat16 and hidden.shape==(1,2048,4096) and torch.isfinite(hidden).all()
   hidden=hidden[0,:-1].to('cuda:0');labels=tokens[row,1:].to('cuda:0')
   best=torch.full((2047,),-float('inf'),device='cuda:0');arg=torch.zeros((2047,),dtype=torch.long,device='cuda:0')
   lse=torch.full_like(best,-float('inf'),dtype=torch.float32);target=torch.empty_like(lse)
   for start in range(0,head.shape[0],4096):
    stop=min(start+4096,head.shape[0]);weight=head[start:stop].to('cuda:0')
    logits=F.linear(hidden,weight).float();assert torch.isfinite(logits).all()
    update_best(logits,start,best,arg)
    lse=torch.logaddexp(lse,torch.logsumexp(logits,dim=-1))
    mask=(labels>=start)&(labels<stop)
    if mask.any():target[mask]=logits[mask,labels[mask]-start]
    del weight,logits
   row_nll=float((lse-target).sum().item());expected=baseline['per_row'][row]['bf16_nll_sum']
   assert abs(row_nll-expected)<=0.001,('BF16 baseline arithmetic mismatch',row,row_nll,expected)
   top1.append(arg.cpu().tolist());nll.append(row_nll)
   del hidden,labels,best,arg,lse,target
   print(json.dumps({'row_complete':row,'elapsed_seconds':time.time()-started}),flush=True)
 assert len(top1)==32 and all(len(x)==2047 for x in top1)
 out.mkdir(parents=True,exist_ok=True)
 save(out/'input_ids.json',tokens.tolist());save(out/'teacher_top1.json',top1)
 receipt={'schema':'glm53-private-serving-reference-v1','state':'TEACHER_REFERENCE_PREPARED_NOT_SERVING_MEASURED','private_recoverable_corpus':True,
  'input_ids_shape':[32,2048],'teacher_top1_shape':[32,2047],'position_contract':'teacher_top1[row][p] predicts input_ids[row][p+1] for p=0..2046; no special-token insertion',
  'fixture_manifest_sha256':FIXTURE,'token_rows_sha256':TOKENS,'teacher_manifest_sha256':TEACHER,'native_head_shard_sha256':HEAD,
  'head_source_candidate_manifest_sha256':ARTIFACT,'head_shape':list(head.shape),'head_dtype':str(head.dtype),
  'input_ids_json_sha256':sha(out/'input_ids.json'),'teacher_top1_json_sha256':sha(out/'teacher_top1.json'),'teacher_row_sha256':[r['sha256'] for r in manifest['records']],
  'arithmetic':'BF16 F.linear on all2047positions, FP32logits, vocabchunk4096; torch.max first index and strict-greater interchunk merge; ascending token IDs win ties',
  'teacher_nll_sums':nll,'teacher_perplexity':math.exp(sum(nll)/65504),'matched_baseline_nll_max_abs_delta':max(abs(nll[i]-baseline['per_row'][i]['bf16_nll_sum']) for i in range(32)),
  'torch':torch.__version__,'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
  'script_sha256':sha(__file__),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.time()-started}
 save(out/'identity.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',type=Path,default=Path('/release'));p.add_argument('--output',type=Path,default=Path('/output'));p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:self_test()
 else:prepare(a)
