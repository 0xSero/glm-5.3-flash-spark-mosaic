"""Proposed CPU-only metadata view: native GLM MTP, not an EXL3 draft.

Mount source at --container-source and output at /mtp in the serving container.
No source tensors are modified. Requires native-mtp-proposal.patch for strict
embedding/head initialization and multimodal proposer metadata.
"""
import argparse, copy, hashlib, json, struct
from pathlib import Path

def prepare(source, output, container_source, native_config=None):
    config_path = native_config or (source / 'source-config.json' if (source / 'source-config.json').is_file() else source / 'config.json')
    config = json.loads(config_path.read_text())
    text = config.get('text_config', config)
    assert text['num_hidden_layers'] == 45
    assert text['num_nextn_predict_layers'] == 1
    # Use the ORIGINAL unpruned source/config for the native MTP companion.
    assert text['n_routed_experts'] == 288
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    keys = {k: v for k, v in index['weight_map'].items() if '.layers.45.' in k or k in ('lm_head.weight', 'model.language_model.embed_tokens.weight')}
    assert len(keys) == 891, f'Expected889 native MTP tensors and2 shared tensors, got{len(keys)}'
    headers = {}
    for file in set(keys.values()):
        assert not Path(file).is_absolute() and ".." not in Path(file).parts
        with (source / file).open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            assert size < 100_000_000
            headers.update(json.loads(f.read(size)))
    for key in keys:
        assert headers[key]['dtype'] in ('BF16', 'F32'), (key, headers[key]['dtype'])
    assert len([k for k in keys if '.mlp.experts.' in k]) == 864
    assert headers['model.language_model.layers.45.mlp.gate.weight']['shape'] == [288,4096]
    assert headers['model.language_model.layers.45.mlp.gate.e_score_correction_bias']['shape'] == [288]
    config = copy.deepcopy(config)
    config.pop('quantization_config', None)
    if 'text_config' in config:
        config['text_config'].pop('quantization_config', None)
    # Keep original glm5_next wrapper so built-in MTP config override is used.
    config['architectures'] = ['Glm5NextForConditionalGeneration']
    config['image_token_index'] = config['image_token_id']
    output.mkdir(parents=True, exist_ok=False)
    assert len({Path(file).name for file in keys.values()}) == len(set(keys.values()))
    for file in set(keys.values()):
        (output / Path(file).name).symlink_to(str(Path(container_source) / file))
    keys = {key: Path(file).name for key, file in keys.items()}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json'):
        if (source / name).is_file():
            (output / name).symlink_to(str(Path(container_source) / name))
    (output / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (output / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': sum(headers[k]['data_offsets'][1]-headers[k]['data_offsets'][0] for k in keys)}, 'weight_map': keys}, indent=2)+'\n')
    receipt = {'native_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(), 'candidate_config_sha256': hashlib.sha256((source/'config.json').read_bytes()).hexdigest(), 'source_index_sha256': hashlib.sha256((source/'model.safetensors.index.json').read_bytes()).hexdigest(), 'native_mtp_tensor_count':889, 'index_tensor_count':891, 'native_mtp_bytes':sum(headers[k]['data_offsets'][1]-headers[k]['data_offsets'][0] for k in keys if '.layers.45.' in k), 'draft_quantization':None, 'mtp_experts':288, 'serving_validated':False}
    (output / 'native-mtp-view.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--container-source',default='/native-source');p.add_argument('--native-config',type=Path);a=p.parse_args();print(json.dumps(prepare(a.source,a.output,a.container_source,a.native_config),indent=2))
