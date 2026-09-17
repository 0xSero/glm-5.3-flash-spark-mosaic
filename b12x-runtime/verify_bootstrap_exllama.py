"""Verify exact reused bootstrap package/extension; no upstream build claim."""
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path

import torch
import exllamav3_ext

pin = json.loads(Path(__file__).with_name('exllama-bootstrap-pin.json').read_text())
assert metadata.version('exllamav3') == pin['version']
assert torch.__version__ == pin['torch'] and torch.version.cuda == pin['cuda']
extension = Path(exllamav3_ext.__file__)
assert hashlib.sha256(extension.read_bytes()).hexdigest() == pin['extension_sha256']
root = Path(importlib.util.find_spec('exllamav3').origin).parent
files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(root.rglob('*')) if p.is_file()
         and '__pycache__' not in p.parts and p.suffix != '.pyc'}
tree = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
assert tree == pin['package_tree_sha256'], 'Bootstrap ExLlama package changed'
for name in ('reconstruct', 'exl3_moe', 'exl3_moe_max_concurrency'):
    assert callable(getattr(exllamav3_ext, name))
assert not torch.cuda.is_initialized()
print(json.dumps({'state': 'PINNED_BOOTSTRAP_EXLLAMA_VERIFIED_GPU_GATE_PENDING',
                  'version': pin['version'], 'extension_sha256': pin['extension_sha256'],
                  'package_tree_sha256': tree, 'cuda_initialized': False,
                  'native_cpu_allreduce_supported_on_arm': False}))
