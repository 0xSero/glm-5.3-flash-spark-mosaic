#!/usr/bin/env python3
"""Read-only pinned K2 identity validator; metadata-only is not a payload seal."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

REPO='0xSero/GLM-5.3-Flash-EXL3-TR3-2.0bpw'
REVISION='35b4b580debe2a1510a2eb042d9d6d4cd7fb3a6b'
INVENTORY_SHA='36b79ee52ace17008eb917588666e3ae7f6ce5f419470df019da8a27576d594b'
BYTES=111352026456


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def checked(root,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe source-relative path')
    f=root/p
    if not f.is_file():raise ValueError(f'missing source file: {name}')
    return f


def validate_file(root,item,full_hash=True):
    f=checked(root,item['path'])
    if f.stat().st_size!=item['bytes']:raise ValueError(f'source byte mismatch: {item["path"]}')
    if full_hash and sha(f)!=item['sha256']:raise ValueError(f'source SHA mismatch: {item["path"]}')


def validate_metadata(root,inventory,bundle):
    for name,expected in inventory['metadata_sha256'].items():
        if sha(checked(bundle,name))!=expected:raise ValueError(f'pinned metadata bundle corrupted: {name}')
        if name!='protected-q3-parity.json' and sha(checked(root,name))!=expected:
            raise ValueError(f'local metadata differs from pinned HF source: {name}; preserve local runtime overrides and create a separate canonical metadata view')
    manifest=json.loads((bundle/'EXL3_MANIFEST.json').read_text())
    if (manifest.get('schema_version')!=2 or manifest.get('state')!='STRUCTURAL_PASS' or
            manifest.get('target_bpw')!=2.0 or manifest.get('effective_routed_tier_bpw')!=2.0 or
            manifest.get('retained_tensor_bytes')!=33835039608 or
            manifest.get('retained_manifest_sha256')!='8d6e5328de1a5f27270167f4fa9c05ea00ac257de8c896f33f8dc7ac4d724b7e'):
        raise ValueError('source is not expected uniform K2 with original protected tensors')
    expected={x['path']:(x['bytes'],x['sha256']) for x in inventory['files']}
    declared={x['path']:(x['bytes'],x['sha256']) for x in manifest['files']}
    if declared!=expected:raise ValueError('HF manifest and LFS inventory disagree')
    config=json.loads((bundle/'config.json').read_text());text=config['text_config'];quant=config['quantization_config']
    if (text['n_routed_experts']!=288 or text['num_hidden_layers']!=45 or text['hidden_size']!=4096 or
            text['n_group']!=1 or text['topk_group']!=1 or text['num_experts_per_tok']!=8 or
            quant['bits']!=2 or quant['k2_experts_per_layer']!=288 or quant['k3_experts_per_layer']!=0 or
            quant['k4_retained_experts_per_layer']!=0):
        raise ValueError('K2 source geometry or bitrate mismatch')
    protected=json.loads((bundle/'protected-q3-parity.json').read_text())
    if (protected['retained_shards']!=49 or protected['retained_tensor_count']!=2482 or
            protected['retained_tensor_bytes']!=33835039608 or not protected['all_retained_shards_identical']):
        raise ValueError('protected parity proof incomplete')
    for item in protected['files']:
        if expected.get(item['path'])!=(item['bytes'],item['sha256']):raise ValueError('protected shard parity mismatch')
    return manifest


def check_trellis(name,value):
    if name.endswith('.trellis') and '.mlp.experts.' in name:
        expected=[32,256,32] if '.down_proj.' in name else [256,32,32]
        if value['dtype']!='I16' or value['shape']!=expected:
            raise ValueError(f'non-K2 or wrong-shape trellis: {name}')
        if value['data_offsets'][1]-value['data_offsets'][0]!=524288:
            raise ValueError('invalid K2 trellis payload size')
        return 1
    return 0


def validate(root,inventory_path,full_hash=True):
    root,inventory_path=Path(root),Path(inventory_path)
    if sha(inventory_path)!=INVENTORY_SHA:raise ValueError('K2 inventory pin mismatch')
    inventory=json.loads(inventory_path.read_text())
    if (inventory.get('schema')!='glm53-pinned-k2-source-inventory-v1' or inventory['repo']!=REPO or
            inventory['revision']!=REVISION or inventory['weight_file_count']!=133 or inventory['weight_file_bytes']!=BYTES):
        raise ValueError('K2 source identity mismatch')
    manifest=validate_metadata(root,inventory,inventory_path.parent)
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    paths=[x['path'] for x in inventory['files']]
    if len(set(paths))!=133 or set(index.values())!=set(paths) or len(index)!=583090:
        raise ValueError('source index file/tensor closure mismatch')
    seen=set();trellises=0
    for item in inventory['files']:
        validate_file(root,item,full_hash)
        f=checked(root,item['path'])
        with f.open('rb') as h:
            n=struct.unpack('<Q',h.read(8))[0]
            if not 2<=n<=64<<20:raise ValueError('invalid header length')
            header=json.loads(h.read(n))
        for key,value in header.items():
            if key=='__metadata__':continue
            if key in seen or index.get(key)!=item['path']:raise ValueError('source header/index tensor mismatch')
            seen.add(key);trellises+=check_trellis(key,value)
    if seen!=set(index) or trellises!=42*288*3*4:raise ValueError('K2 expert field closure incomplete')
    return {'schema':'glm53-k2-source-validation-v1',
            'state':'FULL_PAYLOAD_VERIFIED' if full_hash else 'METADATA_HEADERS_AND_SIZES_ONLY',
            'full_payload_sha256_verified':full_hash,'source_repo':REPO,'source_revision':REVISION,
            'inventory_sha256':INVENTORY_SHA,'weight_files':133,'weight_file_bytes':BYTES,
            'indexed_tensors':len(index),'k2_trellises':trellises,'native_protected_identical_to_q3':full_hash,
            'manifest_sha256':sha(root/'EXL3_MANIFEST.json'),
            'metadata_sha256':inventory['metadata_sha256'],
            'quality_claim':False,'runtime_claim':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--metadata-only',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('use a fresh receipt file')
    result=validate(a.source,a.inventory,not a.metadata_only)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
