import contextlib
import fcntl
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from seal_glm_observations import add_observations, candidate_maps, digest_file, seal
from reap.glm53_record_observation_pipeline import record_contract, make_receipt
from reap.glm53_prepare_record_tokens import record_identity, SOURCE_SHA256, TOKENIZER_SHA256


class SealTests(unittest.TestCase):
    def fixture(self, base):
        token = base / 'tokens'
        token.mkdir()
        rows = [{'input_ids': [1], 'original_tokens': 1, 'source_row': i, 'id': str(i),
                 'truncated': False, 'domain': 'coding'} for i in range(64)]
        (token / 's0.json').write_text(json.dumps({'schema': 'glm53-record-token-shard-v1', 'records': rows}))
        shard = {'shard_id': 0, 'start': 0, 'end': 64, 'sequence_count': 64, 'tokens': 64,
                 'lengths': [1] * 64, 'path': 's0.json', 'sha256': digest_file(token / 's0.json'),
                 'source_rows': list(range(64)), 'record_ids': [str(i) for i in range(64)]}
        final = {**shard, 'shard_id': 1, 'start': 64, 'end': 65, 'sequence_count': 1,
                 'tokens': 1, 'lengths': [1], 'path': 's1.json'}
        manifest = {'schema': 'glm53-record-token-manifest-v1', 'state': 'COMPLETE', 'sequence_length': 16384,
                    'corpus': 'ours', 'identity': record_identity(SOURCE_SHA256, TOKENIZER_SHA256),
                    'sequence_count': 65, 'tokens': 65, 'shards': [shard, final],
                    'domain_counts': {'coding': 64, 'science': 1}}
        mp = token / 'manifest.json'
        mp.write_text(json.dumps(manifest))
        model = {'name': 'test/model', 'revision': 'abc', 'sha256': 'a' * 64}
        modelp = base / 'model.json'
        modelp.write_text(json.dumps(model))
        contract = record_contract(model, manifest, digest_file(mp))
        root = base / contract['run_id']
        root.mkdir()
        (root / 'pipeline.lock').touch()
        runtime = {'prefill_policy': 'test', 'prefill_chunk_size': 1,
                   'kda_implementation': {'module': 'test', 'function': 'test', 'package_version': '1'},
                   'expert_implementation': {'backend': 'portable', 'adapter_sha256': 'b' * 64}}
        row = {'expert_frequency': [64] * 8 + [0] * 280, 'total_tokens': 64}
        for key in ('ean_sum', 'weighted_ean_sum', 'weighted_ean_model_scaled_sum',
                    'weighted_expert_frequency_sum', 'max_activations'):
            row[key] = [1.] * 8 + [0.] * 280
        obs = {str(i): row for i in range(3, 45)}
        receipt = make_receipt(contract, shard, obs, runtime)
        sp = root / 'shard-00000'
        sp.mkdir()
        (sp / 'receipt.json').write_text(json.dumps(receipt))
        (root / 'record-status.json').write_text(json.dumps({'state': 'PAUSED_AT_RECORD_SHARD',
            'run_id': contract['run_id'], 'completed_shards': [0], 'records': 64, 'tokens': 64}))
        return SimpleNamespace(token_manifest=mp, model_identity=modelp, run_root=root,
                               output=base / 'sealed', keep=[8], seed=42)

    def test_partial_seal_preserves_missing_domain(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp).resolve())
            with contextlib.redirect_stdout(io.StringIO()):
                seal(args)
            summary = json.loads((args.output / 'seal.json').read_text())
            self.assertEqual(summary['state'], 'SEALED_PARTIAL_ORIGINAL_RECORDS')
            self.assertEqual(summary['missing_domains'], ['science'])
            self.assertFalse(summary['quality_accepted'])

    def test_tampered_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp).resolve())
            path = args.run_root / 'shard-00000/receipt.json'
            d = json.loads(path.read_text())
            d['observations']['3']['weighted_ean_sum'][0] = 2
            path.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                seal(args)
            self.assertFalse(args.output.exists())

    def test_active_observer_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp).resolve())
            with (args.run_root / 'pipeline.lock').open('r') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    seal(args)

    def test_mass_not_conditional_mean(self):
        total = {'3': {'weighted_ean_sum': [100., 10.], 'expert_frequency': [100, 1]}}
        domains = {'code': total}
        candidates = candidate_maps(total, domains, [1])
        self.assertEqual(candidates['mass_global']['3']['keep_maps']['1']['keep'], [0])
        self.assertEqual(candidates['conditional_mean_control']['3']['keep_maps']['1']['keep'], [1])

    def test_massmax_retains_domain_specialist(self):
        total = {'3': {'weighted_ean_sum': [900., 101.], 'expert_frequency': [900, 101]}}
        domains = {'code': {'3': {'weighted_ean_sum': [900., 100.]}},
                   'math': {'3': {'weighted_ean_sum': [0., 1.]}}}
        candidates = candidate_maps(total, domains, [1])
        self.assertEqual(candidates['massmax_domain']['3']['keep_maps']['1']['keep'], [1])


if __name__ == '__main__':
    unittest.main()
