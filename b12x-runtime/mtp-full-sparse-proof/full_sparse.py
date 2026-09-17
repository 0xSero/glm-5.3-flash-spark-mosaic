"""Full native MTP loader plus sparse288-expert correctness; never a serving/speed test."""
import argparse,contextlib,hashlib,json,os,runpy
from pathlib import Path
W=Path(__file__).resolve().parent
FIELDS=('w13_weight','w13_weight_scale','w13_weight_scale_2','w2_weight','w2_weight_scale','w2_weight_scale_2')
BOUNDARY=(0,7,8,63,127,128,255,287)
LIMIT=32*2**30

def freeze_check():
 pins=json.loads((W/'SOURCE_FREEZE.json').read_text())
 for name,r in pins.items():assert hashlib.sha256((W/name).read_bytes()).hexdigest()==r['sha256'],name
 return pins

@contextlib.contextmanager
def capture_raw_setup(cls,raw):
 """Test-local hook. Preserve the inherited method, including failure restoration."""
 owned='_setup_kernel' in cls.__dict__;original=cls._setup_kernel
 def observed(self,layer):
  if raw:raise ValueError('unexpected second expert setup')
  raw.update({n:getattr(layer,n).detach().cpu().clone() for n in FIELDS})
  return original(self,layer)
 cls._setup_kernel=observed
 try:yield
 finally:
  assert cls._setup_kernel is observed,'setup hook changed concurrently'
  if owned:cls._setup_kernel=original
  else:delattr(cls,'_setup_kernel')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--cpu-import-check',action='store_true');a=parser.parse_args();pins=freeze_check()
 import torch
 import torch.nn.functional as F
 import glm_mtp_expert_precision as policy
 from packed_routes import LaunchRecorder
 from reference_packed import reference_packed,SOURCE_SHA256
 if a.cpu_import_check:
  assert not torch.cuda.is_initialized()
  # Import the new policy directly; do not install its draft dispatcher.
  from policy import enable_for_probe
  assert enable_for_probe is policy.maybe_enable_mtp_expert_precision
  print(json.dumps({'state':'CPU_IMPORT_ONLY_PASS','cuda_initialized':False,'source_pins':pins}));return
 assert os.environ['GLM53_DRAFT_PROBE_MODE']=='nvfp4' and os.environ['GLM53_MTP_EXPERT_NVFP4']=='1'
 assert os.environ['GLM53_MTP_EXPERT_FP8']=='0'
 out=Path('/out');out.mkdir(exist_ok=True)
 report={'state':'STARTED_NOT_ACCEPTED','source_pins':pins,'cells':[],'expert_count':288,'speed_claim':False,'target_or_kv_loaded':False,'serving_acceptance':False}
 def save():
  tmp=out/'full-sparse.json.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'full-sparse.json')
 save();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(1947)
 raw={}
 with capture_raw_setup(policy.B12xNvfp4DraftMethod,raw):ns=runpy.run_path(str(W/'full_load.py'),run_name='__full_native_draft_proof__')
 # The immutable loader's strict native audit/head comparison has completed.
 ns['eagle_utils'].get_model=ns['original_get'];ns['DefaultModelLoader'].track_weights_loading=ns['original_track']
 assert len(raw)==6 and raw['w13_weight'].shape==(288,4096,2048) and raw['w2_weight'].shape==(288,4096,1024)
 assert raw['w13_weight'].dtype==raw['w2_weight'].dtype==torch.uint8
 assert raw['w13_weight_scale'].dtype==raw['w2_weight_scale'].dtype==torch.float8_e4m3fn
 assert all(v.device.type=='cpu' for v in raw.values())
 report['raw_quantized_cpu_bytes']=sum(v.numel()*v.element_size() for v in raw.values())
 report['raw_quantized_metadata']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in raw.items()}
 report['full_native_load_receipt_sha256']=hashlib.sha256((out/'full-load.json').read_bytes()).hexdigest()
 draft=ns['model'];c=ns['c'];moe=draft.get_submodule('model.layers.45.mtp_block.mlp');routed=moe.experts.routed_experts;method=routed.quant_method;kernel=method.moe_kernel.fused_experts
 assert isinstance(kernel,policy.PackedDraftB12xExperts) and kernel._quant_mode=='w4a16'
 cfg=c.speculative_config.draft_model_config.hf_config
 assert cfg.n_routed_experts==288 and cfg.num_experts_per_token==8 and cfg.n_group==1 and cfg.scoring_func=='sigmoid' and cfg.moe_renormalize and cfg.swiglu_limit==10
 from safetensors import safe_open
 index=json.loads(Path('/mtp/model.safetensors.index.json').read_text())['weight_map']
 def native(name):
  with safe_open('/mtp/'+index[name],framework='pt',device='cpu') as f:return f.get_tensor(name)
 prefix='model.language_model.layers.45.mlp.'
 assert torch.equal(moe.gate.weight.cpu(),native(prefix+'gate.weight'))
 assert torch.equal(moe.gate.e_score_correction_bias.cpu(),native(prefix+'gate.e_score_correction_bias'))
 assert torch.equal(moe.shared_experts.gate_up_proj.weight.cpu(),torch.cat([native(prefix+'shared_experts.'+p+'.weight') for p in ('gate_proj','up_proj')]))
 assert torch.equal(moe.shared_experts.down_proj.weight.cpu(),native(prefix+'shared_experts.down_proj.weight'))
 report['native_router_shared_all_values_exact']=True
 import b12x
 br=Path(b12x.__file__).resolve().parent.parent
 for n,h in SOURCE_SHA256.items():assert hashlib.sha256((br/n).read_bytes()).hexdigest()==h
 from vllm.config import set_current_vllm_config
 from vllm.forward_context import set_forward_context
 from vllm.v1.worker.workspace import init_workspace_manager
 def route(x,forced=None):
  scores=moe.gate(x)[0].float().sigmoid();ids=(scores+moe.gate.e_score_correction_bias.float()).topk(8,-1).indices if forced is None else forced
  weights=scores.gather(1,ids.long());weights=weights/weights.sum(-1,keepdim=True)*cfg.routed_scaling_factor
  return ids.int(),weights
 def relative(a,b):return float(torch.linalg.vector_norm(a.float()-b.float())/torch.linalg.vector_norm(b.float()).clamp_min(1e-12))
 def compare(actual,expected):
  rms=float(expected.float().square().mean().sqrt());torch.testing.assert_close(actual.float(),expected.float(),atol=max(.003,.003*rms),rtol=.03)
  rel=relative(actual,expected);assert rel<.015;return {'relative_l2':rel,'max_abs':float((actual.float()-expected.float()).abs().max())}
 with set_current_vllm_config(c),torch.inference_mode(),LaunchRecorder() as recorder:
  init_workspace_manager(torch.device('cuda:0'))
  with set_forward_context(None,c,num_tokens=8):kernel.warmup_launches(routed,token_counts=(1,2,4,8))
  report['packed_plans']=kernel.packed_plan_metadata();assert all(p['eager_warmup_completed'] and p['route_mode']=='packed' for p in report['packed_plans'])
  assert {p['capacity'] for p in report['packed_plans']}=={1,2,4,8}
  save()
  for tokens,input_scale in ((1,.1),(1,8.),(2,.1),(4,.1),(8,.1)):
   for route_mode in ('native','boundary_fixed'):
    x=torch.randn(tokens,4096,device='cuda',dtype=torch.bfloat16)*input_scale
    fixed=torch.tensor(BOUNDARY,device='cuda',dtype=torch.int64).expand(tokens,-1).clone()
    with set_forward_context(None,c,num_tokens=tokens):
     ids,weights=route(x,None if route_mode=='native' else fixed)
     def forward():
      if route_mode=='native':return moe(x)
      y=method.apply(routed,x,weights,ids,None,None)
      return (y.float()+moe.shared_experts(x).float()).bfloat16()
     before=x.clone();actual=forward().clone();assert torch.equal(x,before)
     shared=moe.shared_experts(x).clone();routed_actual=method.apply(routed,x,weights,ids,None,None).clone()
     ref,diag=reference_packed(x,ids,weights,raw);full_ref=(ref.float()+shared.float()).bfloat16()
     cell={'tokens':tokens,'input_scale':input_scale,'route_mode':route_mode,'original_ids':ids.cpu().tolist(),'full_reference':compare(actual,full_ref),'routed_reference':compare(routed_actual,ref),'oracle_diagnostics':diag,'graph_replays':[]}
     if input_scale==8.:assert min(diag['clamp_hits'].values())>0,diag
     report['cells'].append(cell);save()
     # Normal workspace, actual warmup and unchanged original graph tolerance.
     stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
     with torch.cuda.stream(stream):
      for _ in range(3):forward()
     torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):output=forward()
     initial_ids=ids.clone()
     for step,scale in enumerate((.05,.1,.5),1):
      x.copy_(torch.randn_like(x)*scale)
      changed_ids,changed_weights=route(x,None if route_mode=='native' else (fixed+step)%288)
      # Fixed-route graph reads these original static allocations.
      if route_mode!='native':ids.copy_(changed_ids);weights.copy_(changed_weights)
      expected=forward().clone();graph.replay();torch.cuda.synchronize()
      row={'input_scale':scale,'routes_changed':not torch.equal(changed_ids,initial_ids),'max_abs':float((output-expected).abs().max()),'relative_l2':relative(output,expected),'bitwise_equal':torch.equal(output,expected)}
      cell['graph_replays'].append(row);save()
      assert row['routes_changed'];assert torch.isfinite(output).all()
      torch.testing.assert_close(output,expected,atol=.001,rtol=.01)
     cell['state']='REFERENCE_AND_CHANGED_ROUTE_GRAPH_PASS';report['physical_launches']=recorder.receipt();save()
     del graph,output,expected,actual,shared,ref,full_ref,routed_actual,before
  report['physical_launches']=recorder.receipt()
  assert recorder.calls and all(r['state']=='RETURNED' and r['returned']['tc_decode_fused_sum'] is False for r in recorder.calls)
  report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();assert report['peak_cuda_allocated_bytes']<=LIMIT
  report['resident_cuda_allocated_bytes']=torch.cuda.memory_allocated();report['state']='FULL288_SPARSE_MOE_REFERENCE_GRAPH_PASS_SERVING_PENDING';save()
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
