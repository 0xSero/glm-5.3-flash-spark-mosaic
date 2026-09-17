import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

MODULE = Path(__file__).parents[1] / 'build_reap_checkpoint.py'
spec = importlib.util.spec_from_file_location('pruner', MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def write_tensor_file(path, tensors):
    header, body = {}, bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(body), len(body) + len(data)]}
        body.extend(data)
    raw = json.dumps(header).encode()
    raw += b' ' * (-len(raw) % 8)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + body)


def payload(path, key):
    header, base = m.read_header(path)
    lo, hi = header[key]['data_offsets']
    with path.open('rb') as f:
        f.seek(base + lo)
        return f.read(hi - lo)


class PruningTests(unittest.TestCase):
    def test_lossless_all_rank_fragments_native_mtp_and_router_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp)/'source', Path(tmp)/'candidate'
            tensors = {}
            for expert in (3, 5, 200):
                for rank in range(4):
                    for proj in ('gate_proj', 'up_proj', 'down_proj'):
                        for field in ('mcg', 'trellis', 'suh', 'svh'):
                            key = f'model.language_model.layers.3.mlp.experts.{expert}.{proj}.rank{rank}.{field}'
                            data = bytes([expert, rank, len(proj), len(field)])
                            tensors[key] = ('U8', [4], data)
            gate = 'model.language_model.layers.3.mlp.gate.weight'
            bias = 'model.language_model.layers.3.mlp.gate.e_score_correction_bias'
            tensors[gate] = ('BF16', [288, 2], b''.join(struct.pack('<HH', i, i+1) for i in range(288)))
            tensors[bias] = ('F32', [288], b''.join(struct.pack('<f', i+.25) for i in range(288)))
            native = 'model.language_model.layers.45.mlp.experts.287.down_proj.weight'
            # Noncanonical NaN payload and negative zero prove no float roundtrip.
            tensors[native] = ('BF16', [2], bytes.fromhex('c17f0080'))
            tensors['model.visual.test.weight'] = ('F32', [1], bytes.fromhex('0100c07f'))
            write_tensor_file(src, tensors)
            original = src.read_bytes()
            receipt, emitted = m.rewrite_shard(src, dst, {3: {5: 0, 200: 1}})
            self.assertEqual(src.read_bytes(), original)
            self.assertEqual(receipt['counts'], {'routed': 96, 'router_sliced': 2, 'native_unchanged': 2})
            for old in (5, 200):
                for rank in range(4):
                    for proj in ('gate_proj', 'up_proj', 'down_proj'):
                        for field in ('mcg', 'trellis', 'suh', 'svh'):
                            key = f'model.language_model.layers.3.mlp.experts.{old}.{proj}.rank{rank}.{field}'
                            target = key.replace(f'.experts.{old}.', f'.experts.{0 if old==5 else 1}.')
                            self.assertEqual(payload(src, key), payload(dst, target))
            self.assertEqual(payload(dst, gate), struct.pack('<HHHH', 5, 6, 200, 201))
            self.assertEqual(payload(dst, bias), struct.pack('<ff', 5.25, 200.25))
            self.assertEqual(payload(dst, native), bytes.fromhex('c17f0080'))
            self.assertEqual(m.digest(dst), receipt['sha256'])
            self.assertFalse(dst.with_name(dst.name+'.tmp').exists())
            self.assertEqual(len(emitted), 100)

    def test_full_keep_is_tensor_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp)/'source', Path(tmp)/'candidate'
            key = 'model.language_model.layers.3.mlp.gate.e_score_correction_bias'
            write_tensor_file(src, {key: ('F32', [288], b'1234'*288)})
            m.rewrite_shard(src, dst, {3: dict(enumerate(range(288)))})
            self.assertEqual(payload(src, key), payload(dst, key))

    def test_duplicate_keep_and_incomplete_partition_rejected(self):
        entry = {'keep': list(range(8)), 'prune': list(range(8,288))}
        candidate = {'schema':'glm53-original-records-candidates-v1',
                     'state':'CANDIDATES_NOT_QUALITY_ACCEPTED',
                     'selection':{'mass_global':{'3':{'keep_maps':{'8':entry}}}}}
        self.assertEqual(len(m.selection_maps(candidate,'mass_global',8,layers=[3])[3]),8)
        entry['keep'][-1] = 6
        with self.assertRaises(ValueError):m.selection_maps(candidate,'mass_global',8,layers=[3])
        entry['keep'][-1] = 7
        entry['prune'].pop()
        with self.assertRaises(ValueError):m.selection_maps(candidate,'mass_global',8,layers=[3])

    def test_explicit_source_pins_and_inventory_confusion(self):
        import copy
        root = MODULE.parent
        q3_path = root / 'source-inventory/filehash-inventory.json'
        k2_path = root / 'source-inventory-k2/filehash-inventory.json'
        q3, k2 = json.loads(q3_path.read_text()), json.loads(k2_path.read_text())
        self.assertEqual(m.source_profile(q3, q3_path)['bits'], 3)
        self.assertEqual(m.source_profile(k2, k2_path)['bits'], 2)
        with self.assertRaisesRegex(ValueError, 'content'):
            m.source_profile(k2, q3_path)
        for field in ('repo', 'revision', 'manifest_sha256', 'weight_file_bytes'):
            broken = copy.deepcopy(k2)
            broken[field] = q3[field]
            with self.assertRaisesRegex(ValueError, 'pin'):
                m.source_profile(broken)

    def test_bitrate_metadata_and_native_contract(self):
        import copy
        for dirname, bits, keep in [('source-inventory', 3, 176), ('source-inventory-k2', 2, 256)]:
            config = json.loads((MODULE.parent / dirname / 'config.json').read_text())
            original = copy.deepcopy(config)
            quant = m.candidate_quantization(config, bits, keep)
            self.assertEqual(config, original)
            self.assertEqual(quant['bits'], bits)
            self.assertEqual(quant['native_mtp_n_routed_experts'], 288)
            self.assertEqual(quant['archive_tensor_parallel_size'], 4)
            self.assertEqual(quant['runtime_tensor_parallel_size'], 1)
            self.assertIn(f'0..{keep-1}', quant['quantized_scope'])
            if bits == 2:
                self.assertEqual(quant['k2_experts_per_layer'], keep)
                self.assertEqual(quant['tail_experts_per_layer'], keep)
                self.assertEqual(quant['effective_routed_tier_bpw'], 2.0)
            else:
                self.assertEqual(quant['trellis_k'], 3)
                self.assertEqual(quant['bits_per_weight'], 3.0)

    def test_k2_trellis_exact_shape_and_bits_survive_remap(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp)/'source', Path(tmp)/'candidate'
            tensors = {}
            for rank in range(4):
                key = f'model.language_model.layers.3.mlp.experts.27.down_proj.rank{rank}.trellis'
                tensors[key] = ('I16', [32, 256, 32], bytes(range(256))*2048)
            write_tensor_file(src, tensors)
            original = m.digest(src)
            receipt, _ = m.rewrite_shard(src, dst, {3: {27: 0}})
            self.assertEqual(m.digest(src), original)
            self.assertEqual(receipt['counts'], {'routed': 4})
            header, _ = m.read_header(dst)
            for name in tensors:
                target = name.replace('.experts.27.', '.experts.0.')
                self.assertEqual(header[target]['shape'], [32, 256, 32])
                self.assertEqual(payload(src, name), payload(dst, target))

    def test_bad_router_shape_fails_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp)/'source', Path(tmp)/'candidate'
            write_tensor_file(src, {'model.language_model.layers.3.mlp.gate.weight': ('F32',[287,1],b'1234'*287)})
            with self.assertRaises(ValueError):m.rewrite_shard(src,dst,{3:{0:0}})
            self.assertFalse(dst.exists())

    def test_omitted_empty_shard_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp)/'source', Path(tmp)/'candidate'
            write_tensor_file(src, {'model.language_model.layers.3.mlp.experts.2.gate_proj.rank0.mcg': ('I32',[],b'1234')})
            receipt, emitted = m.rewrite_shard(src,dst,{3:{3:0}})
            self.assertTrue(receipt['omitted_empty_shard'])
            self.assertEqual(emitted,{})
            self.assertFalse(dst.exists())


if __name__ == '__main__':unittest.main()
