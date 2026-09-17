"""Execute actual pinned loader methods with real CPU/meta torch; no GPU."""
import ast, hashlib, json, re, sys, types
from pathlib import Path
import torch
p=Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py')
s=p.read_text();tree=ast.parse(s)
ns={'torch':torch,'Parameter':torch.nn.Parameter,'RANK_STACKED_TP':4,'RANK_STACKED_INTERMEDIATE':512,'EXL3_SUFFIXES':('trellis','suh','svh','mcg'),'_RANK_IN_NAME':re.compile(r'(?:^|[._])rank(?P<rank>\d+)(?=[._])'),'set_weight_attrs':lambda p,d:[setattr(p,k,v) for k,v in d.items()]}
def extract(nodes,names,namespace):
 for n in nodes:
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names:
   n.decorator_list=[]
   m=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,str(p),'exec'),namespace)
extract(tree.body,['map_topk_to_local','expand_rank_stacked_routing','_suffix_from_mapped_name','_rank_from_mapped_name'],ns)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Exl3MoEMethod')
extract(cls.body,['create_weights','_load_exl3'],ns)
dist=types.ModuleType('vllm.distributed');dist.get_tensor_model_parallel_rank=lambda:0;dist.get_tensor_model_parallel_world_size=lambda:1;sys.modules['vllm.distributed']=dist
rows=[]
for count in (176,192):
 for bits in (2,3):
  method=types.SimpleNamespace(bits=bits,quant_config=types.SimpleNamespace(rank_stacked_tp=4,raw_config={}))
  method._load_exl3=types.MethodType(ns['_load_exl3'],method)
  layer=torch.nn.Module();layer._map_global_expert_id_to_local_expert_id=lambda x:x
  with torch.device('meta'):ns['create_weights'](method,layer,count,4096,2048,torch.bfloat16)
  params=dict(layer.named_parameters());assert len(params)==32
  for r in range(4):
   assert tuple(params[f'w13_rank{r}.trellis'].shape)==(count,2,256,32,bits*16)
   assert tuple(params[f'w2_rank{r}.trellis'].shape)==(count,32,256,bits*16)
  ids=torch.tensor([[0,1,2,3,4,5,6,count-1],[-1,count,0,0,0,0,0,0]])
  weights=torch.arange(16,dtype=torch.float32).reshape(2,8)/17
  ex,ew=ns['expand_rank_stacked_routing'](ids,weights,count,4,None)
  assert ex.shape==(2,32) and ew.shape==(2,32)
  assert ex[0,-4:].tolist()==list(range((count-1)*4,count*4))
  assert ex[1,:8].tolist()==[count*4]*8
  assert torch.equal(ew,weights.repeat_interleave(4,dim=1))
  for rank in range(4):
   # Exercise actual copy loader at final valid expert, original native scale length.
   param=torch.nn.Parameter(torch.zeros(count,2,4096,dtype=torch.float16),requires_grad=False)
   param._exl3_owner=layer;param._exl3_archive_rank=rank
   v=torch.arange(4096,dtype=torch.float16)
   assert method._load_exl3(param,v,f'model.layers.3.mlp.experts.routed_experts.w13_rank{rank}.suh','w3',count-1,True)
   assert torch.equal(param[count-1,1],v)
  rows.append({'experts':count,'bits':bits,'meta_parameters':len(params),'virtual_experts':count*4,'expanded_topk':32,'last_expert_copy_pass':True})
# Exercise actual runtime expert name mapper for both candidate sizes.
rp=p.parents[1]/'fused_moe/routed_experts.py';rt=ast.parse(rp.read_text());rc=next(n for n in rt.body if isinstance(n,ast.ClassDef) and n.name=='RoutedExperts')
mn={'EplbState':types.SimpleNamespace(build_initial_global_physical_to_logical_map=lambda n,r:list(range(n))+list(range(r)))}
extract(rc.body,['build_expert_params_mapping'],mn)
for count in (176,192):
 mapping=mn['build_expert_params_mapping']('gate_proj','down_proj','up_proj',count)
 assert len(mapping)==count*3
 for e in (0,count-1):
  for proj,shard in [('gate_proj','w1'),('up_proj','w3'),('down_proj','w2')]:
   raw=f'model.layers.3.mlp.experts.{e}.{proj}.rank3.trellis'
   matches=[x for x in mapping if x[1] in raw];assert len(matches)==1
   target,needle,expert_id,sid=matches[0];assert expert_id==e and sid==shard
   mapped=raw.replace(needle,target);assert mapped.endswith(('w2_rank3.trellis' if shard=='w2' else 'w13_rank3.trellis'))
# Actual grouped-router CPU semantic check with one group and retained bias.
gp=p.parents[1]/'fused_moe/router/grouped_topk_router.py'
gn={'torch':torch,'envs':types.SimpleNamespace(VLLM_USE_FUSED_MOE_GROUPED_TOPK=False,VLLM_BATCH_INVARIANT=True)}
extract(ast.parse(gp.read_text()).body,['grouped_topk'],gn)
torch.manual_seed(722)
logits=torch.randn(11,288);bias=torch.randn(288)*.03;hidden=torch.zeros(11,4096)
for count in (176,192):
 keep=torch.randperm(288)[:count].sort().values
 cw,ci=gn['grouped_topk'](hidden,logits[:,keep],8,True,1,1,'sigmoid',2.5,bias[keep])
 masked_bias=torch.full_like(bias,float('-inf'));masked_bias[keep]=bias[keep]
 rw,ri=gn['grouped_topk'](hidden,logits,8,True,1,1,'sigmoid',2.5,masked_bias)
 assert torch.equal(keep[ci.long()],ri.long())
 assert torch.equal(cw,rw)
 assert torch.allclose(cw.sum(-1),torch.full((11,),2.5))
print(json.dumps({'source_sha256':hashlib.sha256(s.encode()).hexdigest(),'gpu_used':False,'actual_methods_extracted_from_installed_image':True,'allocation_device':'meta','copy_and_routing_device':'cpu','cases':rows,'parameter_mapper_pass':True,'single_group_pruning_semantics_exact':True},indent=2))
