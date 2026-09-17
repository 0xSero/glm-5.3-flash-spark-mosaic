import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import evaluate_pruned_quality as q


class QualityContractTests(unittest.TestCase):
    def artifact(self, root):
        root = Path(root)
        (root / 'config.json').write_text(json.dumps({'text_config': {
            'n_routed_experts': 8, 'num_hidden_layers': 45, 'hidden_size': 4096}}))
        (root / 'data.safetensors').touch()
        weights = {f'model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.rank{rank}.{suffix}': 'data.safetensors'
                   for layer in range(3, 45) for expert in range(8)
                   for proj in ('gate_proj', 'up_proj', 'down_proj') for rank in range(4)
                   for suffix in ('suh', 'svh', 'trellis', 'mcg')}
        (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weights}))
        return weights

    def test_dynamic_expert_count_and_mtp_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            weights = self.artifact(temp)
            weights['model.language_model.layers.45.mlp.experts.287.gate_proj.weight'] = 'data.safetensors'
            (Path(temp) / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weights}))
            count, _ = q.index_contract(Path(temp))
            self.assertEqual(count, 8)

    def test_missing_rank_tensor_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            weights = self.artifact(temp)
            del weights[next(iter(weights))]
            (Path(temp) / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weights}))
            with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
                q.index_contract(Path(temp))

    def test_foreign_expert_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            weights = self.artifact(temp)
            weights['model.language_model.layers.3.mlp.experts.288.gate_proj.rank0.mcg'] = 'data.safetensors'
            (Path(temp) / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weights}))
            with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
                q.index_contract(Path(temp))

    def test_index_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                q.checked_path(Path(temp), '../other.safetensors')

    def test_worker_configures_child_before_original_worker(self):
        calls = []
        def configure():
            calls.append('configured')
            q._worker = lambda *args: ('original_worker', args)
        with patch.object(q, 'configured', configure):
            self.assertEqual(q.layer_worker('variant', 3), ('original_worker', ('variant', 3)))
        self.assertEqual(calls, ['configured'])

    def structural_fixture(self, root):
        self.artifact(root)
        for name in ('BUILD_CONTRACT.json', 'PROTECTED_TENSORS.json', 'quantization_config.json', 'source-config.json'):
            (root / name).write_text('{}')
        metadata = {name: q.sha(root / name) for name in ('BUILD_CONTRACT.json', 'PROTECTED_TENSORS.json',
            'quantization_config.json', 'source-config.json', 'config.json', 'model.safetensors.index.json')}
        manifest = {'schema': 'glm53-physical-reap-exl3-v1', 'state': 'STRUCTURAL_PASS',
                    'keep': 8, 'native_mtp_n_routed_experts': 288, 'metadata_sha256': metadata,
                    'files': [{'path': 'data.safetensors', 'bytes': 0, 'sha256': q.sha(root / 'data.safetensors')}]}
        (root / 'EXL3_MANIFEST.json').write_text(json.dumps(manifest))
        (root / 'BUILD_STATUS.json').write_text(json.dumps({'state': 'COMPLETE',
            'manifest_sha256': q.sha(root / 'EXL3_MANIFEST.json')}))

    def test_candidate_file_hash_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.structural_fixture(root)
            self.assertEqual(q.validate_candidate_files(root, 8), q.sha(root / 'EXL3_MANIFEST.json'))
            (root / 'data.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'payload hash mismatch'):
                q.validate_candidate_files(root, 8)

    def test_changed_config_rejected_despite_stale_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.structural_fixture(root)
            (root / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'metadata hash mismatch'):
                q.validate_candidate_files(root, 8)

    def test_original_cannot_use_pruned_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'EXL3_MANIFEST.json').write_text(json.dumps({'schema_version': 1, 'state': 'STRUCTURAL_PASS'}))
            with self.assertRaisesRegex(ValueError, 'original source requires'):
                q.validate_candidate_files(root, 288)

    def test_pruned_cannot_use_original_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.structural_fixture(root)
            with self.assertRaisesRegex(ValueError, 'all288 experts'):
                q.validate_original_files(root, 8, root / 'EXL3_MANIFEST.json')

    def test_arbitrary_inventory_cannot_enable_original_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'inventory.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'pinned HF inventory'):
                q.validate_original_files(root, 288, root / 'inventory.json')

    def test_original_verifies_all130_files_and_metadata(self):
        # Tiny payloads exercise the full inventory loop; only fixture byte size
        # and inventory seal are patched. Production uses hard-pinned149GB inventory.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = []
            for i in range(130):
                name = f'weight-{i}.safetensors'
                (root / name).write_bytes(b'x')
                files.append({'path': name, 'bytes': 1, 'sha256': q.sha(root / name)})
            (root / 'retained').mkdir()
            for name in ('config.json', 'quantization_config.json', 'retained/manifest.json'):
                (root / name).write_text('{}')
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {str(i): f['path'] for i, f in enumerate(files)}}))
            manifest = {'schema_version': 1, 'state': 'STRUCTURAL_PASS', 'target_bpw': 3.0,
                        'quantized_scope': {'experts_per_layer': 288}}
            (root / 'EXL3_MANIFEST.json').write_text(json.dumps(manifest))
            metadata = {n: q.sha(root / n) for n in ('config.json', 'quantization_config.json',
                'retained/manifest.json', 'model.safetensors.index.json', 'EXL3_MANIFEST.json')}
            inventory = {'schema': 'glm53-pinned-hf-source-inventory-v1', 'repo': q.SOURCE_REPO,
                         'revision': q.SOURCE_REVISION, 'weight_file_count': 130, 'weight_file_bytes': 130,
                         'manifest_sha256': q.sha(root / 'EXL3_MANIFEST.json'), 'metadata_sha256': metadata, 'files': files}
            inv = root / 'inventory.json'
            inv.write_text(json.dumps(inventory))
            with patch.object(q, 'SOURCE_INVENTORY_SHA', q.sha(inv)), patch.object(q, 'SOURCE_BYTES', 130):
                self.assertEqual(q.validate_original_files(root, 288, inv), q.sha(root / 'EXL3_MANIFEST.json'))
                (root / 'weight-129.safetensors').write_bytes(b'y')
                with self.assertRaisesRegex(ValueError, 'original payload hash mismatch'):
                    q.validate_original_files(root, 288, inv)


if __name__ == '__main__':
    unittest.main()
