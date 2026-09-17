#!/usr/bin/env python3
"""Idempotent, pinned Python-only fix for Jovian EXL3 override ordering."""
import argparse,hashlib,json
from pathlib import Path
SOURCE_SHA='1d1429b0536a9ea30d369d1e8f8384732abb18bfad9fd2cde40eeec0eca1ca20'
PATCHED_SHA='da9557e1e46bcf51334757cab5e176cd2d5e2095ef3a601853fafb470f5f4265'
MODEL_CONFIG='vllm/config/model.py'
def digest(data):return hashlib.sha256(data).hexdigest()
def patched_bytes(data):
 if digest(data)==PATCHED_SHA:return data
 if digest(data)!=SOURCE_SHA:raise ValueError('ModelConfig source differs from pinned Jovian; refusing mutation')
 old=b'            overrides = [\n'
 if data.count(old)!=1:raise ValueError('override ordering source anchor mismatch')
 result=data.replace(old,old+b'                "exl3",\n')
 if digest(result)!=PATCHED_SHA:raise ValueError('unexpected ModelConfig patch output')
 compile(result,MODEL_CONFIG,'exec');return result

def apply(path,check_only=False):
 original=path.read_bytes();result=patched_bytes(original)
 if result!=original and not check_only:
  temporary=path.with_name(path.name+'.exl3-config.tmp');temporary.write_bytes(result);temporary.chmod(path.stat().st_mode);temporary.replace(path)
 return {'state':'CONFIG_OVERRIDE_CHECKED' if check_only else 'CONFIG_OVERRIDE_ALREADY_APPLIED' if original==result else 'CONFIG_OVERRIDE_APPLIED_NOT_RUNTIME_VALIDATED','before_sha256':digest(original),'after_sha256':digest(result)}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--model-file',type=Path);g.add_argument('--root',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args()
 print(json.dumps(apply(a.model_file if a.model_file else a.root/MODEL_CONFIG,a.check_only)))
