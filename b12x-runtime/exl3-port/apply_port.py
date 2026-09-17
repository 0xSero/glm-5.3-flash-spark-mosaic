#!/usr/bin/env python3
"""Install the narrow, source-pinned Jovian EXL3 port in a build checkout."""
import argparse
import hashlib
import json
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = 'vllm/model_executor/layers/quantization/__init__.py'
WRAPPER = 'vllm/model_executor/layers/quantization/exl3.py'
MODEL_CONFIG = 'vllm/config/model.py'
_spec=importlib.util.spec_from_file_location('exl3_config_override',HERE/'apply_config_override.py')
_config_patch=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config_patch)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def registry_patch(text):
    for old, new in [
        ('QuantizationMethods = Literal[\n', 'QuantizationMethods = Literal[\n    "exl3",\n'),
        ('    from .experts_int8 import ExpertsInt8Config\n', '    from .experts_int8 import ExpertsInt8Config\n    from .exl3 import Exl3Config\n'),
        ('    method_to_config: dict[str, type[QuantizationConfig]] = {\n',
         '    method_to_config: dict[str, type[QuantizationConfig]] = {\n        "exl3": Exl3Config,\n'),
    ]:
        if text.count(old) != 1:
            raise ValueError('registry source anchor mismatch')
        text = text.replace(old, new)
    return text


def apply(root, check_only=False):
    manifest = json.loads((HERE/'PORT_MANIFEST.json').read_text())
    wrapper = (HERE/'exl3.py').read_bytes()
    if sha(wrapper) != manifest['ported_wrapper_sha256']:
        raise ValueError('ported wrapper changed without refreshed evidence')
    for name, expected in manifest['source_sha256'].items():
        if sha((root/name).read_bytes()) != expected:
            raise ValueError('upstream API source changed: '+name)
    if (root/WRAPPER).exists():
        raise ValueError('refusing to overwrite an existing EXL3 implementation')
    registry = registry_patch((root/REGISTRY).read_text())
    model_config = _config_patch.patched_bytes((root/MODEL_CONFIG).read_bytes())
    if sha(model_config) != manifest['ported_model_config_sha256']:
        raise ValueError('ModelConfig patch differs from manifest')
    compile(wrapper, WRAPPER, 'exec')
    compile(registry, REGISTRY, 'exec')
    if not check_only:
        # All checks precede mutations. Build checkout only, never a live import tree.
        for name, data in [(WRAPPER, wrapper), (REGISTRY, registry.encode()), (MODEL_CONFIG, model_config)]:
            path = root/name
            temporary = path.with_name(path.name+'.exl3-port.tmp')
            temporary.write_bytes(data)
            temporary.replace(path)
    return {'state':'SOURCE_COMPATIBILITY_CHECKED' if check_only else 'PORT_APPLIED_NOT_RUNTIME_VALIDATED',
            'upstream_revision':manifest['upstream_revision'],
            'wrapper_sha256':sha(wrapper), 'registry_sha256':sha(registry.encode()), 'model_config_sha256':sha(model_config)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    print(json.dumps(apply(args.root,args.check_only)))

if __name__=='__main__':
    main()
