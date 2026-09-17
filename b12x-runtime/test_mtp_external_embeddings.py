#!/usr/bin/env python3
"""Actual CPU tensor tests for target-provided image/video MTP embeddings."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings, _require_is_multimodal, supports_multimodal_embeddings,
)
from vllm.model_executor.models.utils import _merge_multimodal_embeddings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, help='Write the same CPU contract receipt as printed.')
parser.add_argument('--source', type=Path, help='Pre-build method extraction; final gate imports the installed model.')
args = parser.parse_args()
if args.source:
    tree = ast.parse(args.source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Glm5NextMTP')
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'embed_input_ids')
    namespace = dict(torch=torch, MultiModalEmbeddings=MultiModalEmbeddings,
                     _require_is_multimodal=_require_is_multimodal,
                     _merge_multimodal_embeddings=_merge_multimodal_embeddings)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(args.source), 'exec'), namespace)
    embed = namespace['embed_input_ids']
else:
    from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP
    assert supports_multimodal_embeddings(Glm5NextMTP)
    embed = Glm5NextMTP.embed_input_ids

weight = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
subject = SimpleNamespace(model=SimpleNamespace(embed_input_ids=lambda ids: weight[ids]))
ids = torch.tensor([1, 2, 3, 4])
assert torch.equal(embed(subject, ids), weight[ids])
assert torch.equal(embed(subject, ids, []), weight[ids])

for label, features in [('image', [torch.arange(8, dtype=torch.float32).reshape(2, 4)+100]),
                        ('video', [torch.full((2, 4), 200.), torch.full((2, 4), 300.)])]:
    count = sum(len(item) for item in features)
    ids = torch.tensor([1] + [999]*count + [2])
    original_ids = ids.clone()
    mask = torch.tensor([False] + [True]*count + [False])
    result = embed(subject, ids, features, is_multimodal=mask)
    assert torch.equal(result[mask], torch.cat(features).to(torch.bfloat16)), label
    assert torch.equal(result[~mask], weight[torch.tensor([1, 2])]), label
    assert torch.equal(ids, original_ids), 'Input IDs were mutated'

try:
    embed(subject, torch.tensor([1, 2]), [torch.zeros(1, 4)])
except ValueError:
    pass
else:
    raise AssertionError('Missing multimodal mask did not fail')
try:
    embed(subject, torch.tensor([1, 2, 3]), [torch.zeros(2, 4)],
          is_multimodal=torch.tensor([True, False, False]))
except ValueError:
    pass
else:
    raise AssertionError('Mismatched feature/placeholder count did not fail')
assert not torch.cuda.is_initialized()
report = {'state': 'CPU_EXTERNAL_EMBEDDING_CONTRACT_PASS',
                  'installed_model_imported': args.source is None,
                  'text_unchanged': True, 'oov_placeholders_masked': True,
                  'image_and_video_feature_positions_exact': True,
                  'invalid_inputs_rejected': True, 'cuda_initialized': False}
if args.output:
    args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
