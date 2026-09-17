#!/usr/bin/env python3
"""Apply only to the verified Jovian EXL3 wrapper, never a running import tree."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
BEFORE='2d62a8bd6f0d2e1df7556ce086f8c294d6c20faa4e4af123300aa595e264584d'
def sha(data):return hashlib.sha256(data).hexdigest()
def patched(source):
    if sha(source)!=BEFORE:raise ValueError('Unexpected EXL3 source; refusing patch')
    text=source.decode()
    for old,new in [
      ('        self.raw_config = dict(kwargs)\n','        self.raw_config = dict(kwargs)\n        from .glm_exl3_layer_bits import parse_layer_bits\n        self.layer_bits = parse_layer_bits(self.raw_config.pop("layer_bits", None))\n'),
      ('            return Exl3MoEMethod(layer.moe_config, self)\n','            from .glm_exl3_layer_bits import for_routed_prefix\n            return Exl3MoEMethod(layer.moe_config, for_routed_prefix(self, prefix))\n')]:
        if text.count(old)!=1:raise ValueError('EXL3 source anchor mismatch')
        text=text.replace(old,new)
    compile(text,'exl3.py','exec')
    return text.encode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wrapper',type=Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    original=(HERE.parent/'exl3-port/exl3.py').read_bytes() if (HERE.parent/'exl3-port/exl3.py').exists() else (HERE/'original_exl3.py').read_bytes()
    after=patched(original);current=a.wrapper.read_bytes();helper=(HERE/'layer_bits.py').read_bytes();dest=a.wrapper.with_name('glm_exl3_layer_bits.py')
    if current not in (original,after):raise ValueError('Unexpected installed wrapper; refusing overwrite')
    if dest.exists() and dest.read_bytes()!=helper:raise ValueError('Unexpected existing layer-bits helper')
    if not a.check_only:
        dest.write_bytes(helper);a.wrapper.write_bytes(after)
    print(json.dumps({'state':'STAGED_SOURCE_CHECK_PASS' if a.check_only else 'PATCHED_GPU_UNQUALIFIED','wrapper_sha256':sha(after),'helper_sha256':sha(helper)}))
if __name__=='__main__':main()
