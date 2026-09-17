#!/usr/bin/env python3
"""CPU import and meta-construction smoke in the exact established observer image."""
import argparse
import inspect
import json
from pathlib import Path
import sys

from evaluate_pruned_quality import prepare_native_imports, sha, BASE_SHA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    wrapper, extension, linear = prepare_native_imports()
    import torch
    import evaluate_low_bpw_quality as baseline
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer
    assert sha(baseline.__file__) == BASE_SHA
    assert baseline._load_linear.__globals__['LinearEXL3'] is linear
    assert baseline.F.linear is torch.nn.functional.linear
    assert callable(extension.reconstruct)
    assert 'flash_attn' not in sys.modules
    assert 'exllamav3.modules.attn' not in sys.modules
    # No Config or attention call in the actual reconstruction methods. The
    # established namespace shim does not replace this real implementation.
    methods = [inspect.getsource(linear.get_weight_tensor), inspect.getsource(linear.get_inner_weight_tensor)]
    assert all('flash_attn' not in source and 'Config(' not in source for source in methods)
    config = Glm5NextConfig.from_pretrained(args.config, local_files_only=True).text_config
    config._attn_implementation = 'sdpa'
    shapes = {}
    with torch.device('meta'):
        for layer_id in (0, 3, 4):
            layer = Glm5NextTextDecoderLayer(config, layer_id)
            shapes[str(layer_id)] = {'block_type': layer.block_type,
                'parameters': sum(p.numel() for p in layer.parameters()),
                'experts': layer.mlp.experts.num_experts if hasattr(layer.mlp, 'experts') else None}
            del layer
    assert not torch.cuda.is_initialized()
    print(json.dumps({'state': 'CPU_IMPORT_AND_META_CONSTRUCTION_PASS',
                      'gpu_execution_tested': False, 'flash_attention_substituted': False,
                      'actual_linear_class': linear.__module__ + '.' + linear.__name__,
                      'baseline_evaluator_sha256': BASE_SHA,
                      'wrapper_sha256': sha(wrapper.__file__),
                      'extension_sha256': sha(extension.__file__),
                      'linear_source_sha256': sha(inspect.getfile(linear)),
                      'torch': torch.__version__, 'meta_layers': shapes}, indent=2))


if __name__ == '__main__':
    main()
