import json,torch
from vllm.model_executor.layers.quantization.exl3 import Exl3Config
from vllm.model_executor.layers.quantization.glm_exl3_layer_bits import for_routed_prefix
c=Exl3Config.from_config({'quant_method':'exl3','bits':2,'rank_stacked_tp':4,'layer_bits':{'5':3,'32':3}})
for n in range(3,45):
 q=for_routed_prefix(c,f'language_model.model.layers.{n}.mlp.experts')
 assert q.bits==(3 if n in (5,32) else 2)
 assert q.rank_stacked_tp==4 and q.raw_config['rank_stacked_tp']==4
assert c.bits==2 and c.layer_bits=={'5':3,'32':3}
assert not torch.cuda.is_initialized()
print(json.dumps({'state':'ACTUAL_EXL3_CONFIG_IMPORT_AND_LAYER_SELECTION_PASS_GPU_UNQUALIFIED','default_bits':c.bits,'layer_bits':c.layer_bits,'target_layers_checked':42,'rank_stacked_tp':4,'shared_config_unchanged':True,'cuda_initialized':False}))
