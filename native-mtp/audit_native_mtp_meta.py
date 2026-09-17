import json,torch,types,struct
from pathlib import Path
from vllm.engine.arg_utils import EngineArgs
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.quantization.exl3 import Exl3Config
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils
from vllm.distributed import init_distributed_environment,ensure_model_parallel_initialized
from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP
c=EngineArgs(model='/model',tokenizer='/model',skip_tokenizer_init=True,dtype='bfloat16',quantization='exl3',max_model_len=262144,max_num_seqs=1,max_num_batched_tokens=2048,kv_cache_dtype='fp8',block_size=64,enable_prefix_caching=False,speculative_config={'method':'mtp','model':'/mtp','num_speculative_tokens':1}).create_engine_config()
print('BASE_QUANT',type(c.quant_config).__name__,flush=True)
# Execute the actual V2 loader used by VLLM_USE_V2_MODEL_RUNNER=1.
# Replace only disk allocation in get_model; constructor and strict key loading stay real.
def get_model_meta(*, vllm_config, model_config):
 global d, model
 d=vllm_config
 assert d.model_config is model_config is c.speculative_config.draft_model_config
 assert d.quant_config is None
 assert d.model_config.hf_config.n_routed_experts == 288
 with set_current_vllm_config(d),torch.device('meta'):
  model=Glm5NextMTP(vllm_config=d)
 return model
eagle_utils.get_model=get_model_meta
init_distributed_environment(world_size=1,rank=0,distributed_init_method='file:///tmp/mtp-meta-dist',local_rank=0,backend='gloo')
with set_current_vllm_config(c):
 ensure_model_parallel_initialized(1,1)
torch.set_default_dtype(torch.bfloat16)
model=eagle_utils.load_eagle_model(types.SimpleNamespace(model=types.SimpleNamespace()),c)
print('PARAMETERS',json.dumps({k:{'shape':list(p.shape),'dtype':str(p.dtype),'device':str(p.device)} for k,p in model.named_parameters()}),flush=True)

# Exact891 native checkpoint keys/shapes passed through the real MTP loader.
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
root=Path('/mtp');idx=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];headers={}
for filename in set(idx.values()):
 with (root/filename).open('rb') as f:
  size=struct.unpack('<Q',f.read(8))[0];headers.update(json.loads(f.read(size)))
dtypes={'BF16':torch.bfloat16,'F32':torch.float32}
def weights():
 for name in sorted(idx):
  h=headers[name];yield name,torch.empty(h['shape'],device='meta',dtype=dtypes[h['dtype']])
with set_current_vllm_config(d):
 loaded=model.load_weights(weights())
 DefaultModelLoader(LoadConfig()).track_weights_loading(model,loaded)
report={'production_loader':'v1.worker.gpu.spec_decode.eagle.utils.load_eagle_model','native_keys':len(idx),'loaded_runtime_parameters':len(loaded),'strict_completeness_pass':True,'draft_quantization':d.quant_config,'draft_experts':d.model_config.hf_config.n_routed_experts,'target_quantization':type(c.quant_config).__name__,'all_weights_meta':all(x.device.type=='meta' for x in model.parameters()),'cuda_allocated_bytes':torch.cuda.memory_allocated()}
Path('/audit/native-mtp-keyclosure-r3.json').write_text(json.dumps(report,indent=2)+'\n');print('KEYCLOSURE',json.dumps(report),flush=True)
