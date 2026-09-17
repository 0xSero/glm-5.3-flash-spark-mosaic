"""Isolated native-router arithmetic/graph probe. Never loads a full model."""
import hashlib,json,os,time,traceback
from pathlib import Path
import torch
from safetensors.torch import load_file
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm
root=Path('/out');root.mkdir(exist_ok=True)
report={'state':'STARTED','full_model_quality_claim':False,'input_kind':'deterministic synthetic BF16; source-exact native router weights','rows':[]}
def save(): (root/'router-gpu.json').write_text(json.dumps(report,indent=2))
save()
try:
 torch.manual_seed(739);torch.cuda.manual_seed_all(739)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 report['allow_bf16_reduced_precision_reduction']=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
 assert torch.cuda.get_device_capability()==(12,1)
 torch.cuda.set_per_process_memory_fraction((512*1024**2)/torch.cuda.get_device_properties(0).total_memory)
 source=load_file('/input/router.safetensors',device='cpu')
 assert source['weight'].shape==(288,4096) and source['weight'].dtype==torch.bfloat16
 assert source['bias'].shape==(288,) and source['bias'].dtype==torch.float32
 report['source_sha256']=hashlib.sha256(Path('/input/router.safetensors').read_bytes()).hexdigest()
 from vllm.distributed import init_distributed_environment,ensure_model_parallel_initialized
 init_distributed_environment(world_size=1,rank=0,distributed_init_method='file:///tmp/router-probe-dist',local_rank=0,backend='gloo')
 ensure_model_parallel_initialized(1,1)
 with torch.device('cuda'):
  gate=GateLinear(4096,288,out_dtype=torch.float32,params_dtype=torch.bfloat16)
 gate.weight.data.copy_(source['weight']);bias=source['bias'].cuda()
 assert not gate.allow_ll_bf16_gemm and not gate.allow_cublas_router_gemm
 report['baseline_flags']={k:getattr(gate,k) for k in ['allow_ll_bf16_gemm','allow_cublas_router_gemm','allow_dsv3_router_gemm']}
 def cublas(x):
  guard=x.new_empty((x.shape[0],288)) if torch.cuda.is_current_stream_capturing() else None
  out=torch.mm(x,gate.weight.T,out_dtype=torch.float32)
  del guard
  return out
 def selection(y):
  s=y.sigmoid();ids=(s+bias).topk(8,dim=-1).indices
  w=s.gather(1,ids);return ids,w/w.sum(-1,keepdim=True)
 def event_us(fn,x):
  # One graph contains20 calls: event timing excludes Python launch gaps.
  timing_graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(timing_graph):
   for _ in range(20):timing_output=fn(x)
  for _ in range(3):timing_graph.replay()
  torch.cuda.synchronize();times=[]
  for _ in range(5):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
   a.record();timing_graph.replay();b.record();b.synchronize()
   times.append(a.elapsed_time(b)*1000/20)
  return sorted(times)[2]
 graphs=[];pool=torch.cuda.graph_pool_handle()
 for m in [1,2,4,5,8,16,17,32,2048]:
  x=torch.randn(m,4096,device='cuda',dtype=torch.bfloat16)*0.1
  reference=x.float()@gate.weight.float().T
  refids,refweights=selection(reference)
  routes={'baseline':lambda a:gate(a)[0],'cublas_fp32':cublas}
  if m<=16:routes['ll_bf16_fp32']=lambda a:ll_bf16_gemm(a,gate.weight)
  for name,fn in routes.items():
   row={'tokens':m,'route':name,'state':'STARTED'};report['rows'].append(row);save()
   try:
    y=fn(x);torch.cuda.synchronize()
    assert y.dtype==torch.float32 and torch.isfinite(y).all()
    ids,w=selection(y)
    err=(y-reference).abs()
    row.update(max_abs_error=float(err.max()),mean_abs_error=float(err.mean()),top8_order_match=float((ids==refids).all(-1).float().mean()),top8_set_match=float((ids.sort(-1).values==refids.sort(-1).values).all(-1).float().mean()),max_rank_aligned_normalized_weight_difference=float((w-refweights).abs().max()))
    if name!='baseline':torch.testing.assert_close(y,reference,atol=0.0005,rtol=0.0005)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
     for _ in range(3):fn(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,pool=pool):gy=fn(x)
    for _ in range(3):graph.replay()
    torch.cuda.synchronize();torch.testing.assert_close(gy,y,atol=0,rtol=0)
    row.update(graph_replays=3,graph_max_abs_error=float((gy-y).abs().max()),median_cudagraph_gpu_us=event_us(fn,x),state='PASS')
    graphs.append((graph,gy,y.clone(),x))
   except Exception as e:row.update(state='FAIL',error=repr(e),traceback=traceback.format_exc())
   save();print(json.dumps(row),flush=True)
 # Shared graph pool replays must follow capture order, retaining every input/output.
 for graph,y,expected,x in graphs:
  graph.replay();torch.cuda.synchronize();torch.testing.assert_close(y,expected,atol=0,rtol=0)
 report['shared_pool_ordered_replay_pass']=True
 report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
 assert report['peak_cuda_allocated_bytes']<1024**3
 report['state']='PASS' if all(r['state']=='PASS' for r in report['rows']) else 'PARTIAL_PATH_FAILURE'
except Exception as e: report.update(state='FAIL',error=repr(e),traceback=traceback.format_exc());raise
finally:save();print(json.dumps(report),flush=True)
