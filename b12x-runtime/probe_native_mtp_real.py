import json,torch,types,struct
from pathlib import Path
from vllm.engine.arg_utils import EngineArgs
from vllm.config import set_current_vllm_config, replace
from vllm.model_executor.layers.quantization.glm_mtp_expert_fp8 import MTPExpertOnlyFp8Config
from vllm.model_executor.layers.quantization.exl3 import Exl3Config
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils
from vllm.distributed import init_distributed_environment,ensure_model_parallel_initialized
from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP
c=EngineArgs(model='/model',tokenizer='/model',skip_tokenizer_init=True,dtype='bfloat16',quantization='exl3',max_model_len=262144,max_num_seqs=1,max_num_batched_tokens=2048,kv_cache_dtype='fp8_ds_mla',attention_backend='B12X',additional_config={'kda_prefill_backend':'b12x'},block_size=256,enable_prefix_caching=False,speculative_config={'method':'mtp','model':'/mtp','num_speculative_tokens':1,'attention_backend':'B12X'}).create_engine_config()
print('BASE_QUANT',type(c.quant_config).__name__,flush=True)
import os
if os.environ.get('GLM53_CONFIG_ONLY') == '1':
 draft=c.speculative_config.draft_model_config
 assert isinstance(c.quant_config,Exl3Config)
 assert draft.hf_config.n_routed_experts==288
 print('ACTUAL_CONFIG_PASS',json.dumps({'target':type(c.quant_config).__name__,'draft_runtime_dispatch':'pending_actual_load','target_experts':c.model_config.hf_config.get_text_config().n_routed_experts,'draft_experts':draft.hf_config.n_routed_experts,'max_model_len':c.model_config.max_model_len,'cache_dtype':c.cache_config.cache_dtype,'cuda_initialized':torch.cuda.is_initialized()}),flush=True)
 raise SystemExit(0)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
original_get_model=eagle_utils.get_model
original_track=DefaultModelLoader.track_weights_loading
tracked=[]
def track(self,model,loaded):
 original_track(self,model,loaded)
 tracked.append({'runtime_parameters':len(dict(model.named_parameters())),'loaded_parameters':len(loaded),'strict_completeness_pass':True})
DefaultModelLoader.track_weights_loading=track
def checked_get_model(*,vllm_config,model_config):
 global d
 d=vllm_config
 assert d.model_config is model_config is c.speculative_config.draft_model_config
 assert isinstance(d.quant_config,MTPExpertOnlyFp8Config) and model_config.hf_config.n_routed_experts==288
 # Observe actual installed dispatch policy; do not replace it in this probe.
 return original_get_model(vllm_config=d,model_config=model_config)
eagle_utils.get_model=checked_get_model
init_distributed_environment(world_size=1,rank=0,distributed_init_method='file:///tmp/mtp-real-dist',local_rank=0,backend='gloo')
with set_current_vllm_config(c):
 ensure_model_parallel_initialized(1,1)
torch.set_default_dtype(torch.bfloat16)
with set_current_vllm_config(c):
 model=eagle_utils.load_eagle_model(types.SimpleNamespace(model=types.SimpleNamespace()),c)
assert tracked and all(p.device.type=='cuda' for p in model.parameters())
assert isinstance(model.quant_config,MTPExpertOnlyFp8Config)
assert len(model.quant_config.selected_prefixes)==1
# Compare native shared embedding/head samples against exact source values.
from safetensors import safe_open
idx=json.loads(Path('/mtp/model.safetensors.index.json').read_text())['weight_map']
comparisons=[]
params=dict(model.named_parameters())
for source_name,target_name in [('model.language_model.embed_tokens.weight','model.embed_tokens.weight'),('lm_head.weight','model.layers.45.shared_head.head.weight')]:
 if source_name not in idx:
  choices=[n for n in idx if n.endswith('embed_tokens.weight' if 'embed_tokens' in source_name else 'lm_head.weight')]
  assert len(choices)==1,choices
  source_name=choices[0]
 with safe_open('/mtp/'+idx[source_name],framework='pt',device='cpu') as f:
  expected=f.get_slice(source_name)[0:4,0:16]
 actual=params[target_name][0:4,0:16].detach().cpu()
 assert torch.equal(expected,actual),(source_name,target_name)
 comparisons.append({'source':source_name,'runtime':target_name,'samples_equal':True})
baseline=json.loads(Path('/audit/native-parameter-baseline.json').read_text())
expert_prefix='model.layers.45.mtp_block.mlp.experts.routed_experts.'
nonexpert_audit={}
for name,expected in baseline.items():
 if name.startswith(expert_prefix):continue
 if name.endswith('indexer.wk_weights_proj.weight'):
  # Jovian split the legacy fused [wk; weights_proj] into two native matrices.
  names=[name.replace('wk_weights_proj','wk'),name.replace('wk_weights_proj','weights_proj')]
  values=[params[n] for n in names]
  assert all(str(v.dtype)==expected['dtype'] for v in values),(names,expected)
  assert [sum(v.shape[0] for v in values),values[0].shape[1]]==expected['shape']
  nonexpert_audit[name]={'runtime_names':names,'dtype':expected['dtype'],'legacy_fused_shape_preserved':True}
  continue
 param=params[name]
 if name.endswith('index_kpool_compress_ape') and str(param.dtype)!=expected['dtype']:
  # Stored source is BF16; legacy allocated an F32 upcast. The new kernel
  # loads APE and converts to F32 before score math. Prove every source value.
  source_names=[n for n in idx if n.endswith('index_kpool_compress_ape')]
  assert len(source_names)==1
  with safe_open('/mtp/'+idx[source_names[0]],framework='pt',device='cpu') as f:
   source=f.get_tensor(source_names[0])
  assert source.dtype==param.dtype==torch.bfloat16 and expected['dtype']=='torch.float32'
  assert torch.equal(source.float(),param.detach().cpu().float())
  nonexpert_audit[name]={'dtype':str(param.dtype),'source_dtype':str(source.dtype),'all_source_values_exact':True,'legacy_f32_upcast_only':True}
  continue
 assert str(param.dtype)==expected['dtype'],(name,str(param.dtype),expected)
 nonexpert_audit[name]={'dtype':str(param.dtype),'matches_legacy_runtime_dtype':True}
expert_audit={name:{'shape':list(param.shape),'dtype':str(param.dtype),'bytes':param.numel()*param.element_size()} for name,param in params.items() if name.startswith(expert_prefix)}
for suffix in ['w13_weight','w2_weight']:
 assert params[expert_prefix+suffix].dtype==torch.float8_e4m3fn
for name,param in params.items():
 if not name.startswith(expert_prefix):assert param.dtype!=torch.float8_e4m3fn,name
report={'probe' :'actual V2 draft-only real-weight load, no target weights or serving forward','production_loader':'v1.worker.gpu.spec_decode.eagle.utils.load_eagle_model','native_index_keys':len(idx),'strict_tracking':tracked,'draft_quantization':'online FP8 only routed experts','selected_prefixes':model.quant_config.selected_prefixes,'native_nonexpert_audit':nonexpert_audit,'expert_parameters':expert_audit,'draft_experts':288,'target_quantization':type(c.quant_config).__name__,'all_parameters_cuda':True,'cuda_allocated_bytes':torch.cuda.memory_allocated(),'source_value_comparisons':comparisons}
Path('/audit/b12x-mtp-expert-fp8-real-load.json').write_text(json.dumps(report,indent=2)+'\n')
print('EXPERT_FP8_REAL_LOAD_PASS',json.dumps(report),flush=True)
