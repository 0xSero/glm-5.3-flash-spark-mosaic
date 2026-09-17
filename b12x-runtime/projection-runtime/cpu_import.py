"""Import installed projection code and parse actual candidate metadata without CUDA."""
import copy
import hashlib
import inspect
import json
from pathlib import Path

import torch
from vllm.model_executor.layers.quantization.exl3 import Exl3Config
from vllm.model_executor.layers.quantization.glm_exl3_projection_bits import resolve_layer_projection_bits

config = json.loads(Path('/candidate/config.json').read_text())
raw = config.get('quantization_config') or config['text_config']['quantization_config']
before = copy.deepcopy(raw)
q = Exl3Config.from_config(raw)
overrides = q.raw_config['layer_projection_bits']
resolved = {str(layer): resolve_layer_projection_bits(q.bits, overrides, layer) for layer in range(3, 45)}
expected = {5, 26, 31, 32, 35, 36}
assert set(map(int, overrides)) == expected
assert all(bits == ((2, 2, 3) if int(layer) in expected else (2, 2, 2)) for layer, bits in resolved.items())
assert raw == before and q.bits == 2 and q.rank_stacked_tp == 4
assert q.get_quant_method(torch.nn.Linear(2, 2, device='meta'), 'language_model.lm_head') is None
assert not torch.cuda.is_initialized()
module = Path(inspect.getfile(Exl3Config))
print(json.dumps({'state': 'CPU_IMPORTED_CANDIDATE_METADATA_PASS',
    'wrapper_sha256': hashlib.sha256(module.read_bytes()).hexdigest(),
    'candidate_config_sha256': hashlib.sha256(Path('/candidate/config.json').read_bytes()).hexdigest(),
    'resolved': resolved, 'cuda_initialized': False,
    'scope': 'Installed imports and actual metadata only; no full model load, GPU allocation, quality or serving claim.'}))
