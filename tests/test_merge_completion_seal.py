import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from seal_glm_observations import digest_file, seal
from merge_completion_seal import main as merge_main
from reap.glm53_record_observation_pipeline import record_contract, make_receipt

# A site-packages 'tests' package shadows this directory as a namespace package,
# so load the fixture class from its file path directly.
_spec = importlib.util.spec_from_file_location(
    'test_seal_glm_observations_fixture', Path(__file__).resolve().parent / 'test_seal_glm_observations.py')
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_FixtureSource = _mod.SealTests  # kept out of TestCase collection by the leading underscore


def run_seal(args_ns):
    with contextlib.redirect_stdout(io.StringIO()):
        seal(args_ns)


def run_merge(argv):
    import sys
    old = sys.argv
    sys.argv = ['merge-completion-seal.py'] + [str(a) for a in argv]
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            merge_main()
        return json.loads(buf.getvalue())
    finally:
        sys.argv = old


class MergeCompletionSealTests(unittest.TestCase):
    def partial_fixture(self, base):
        """Adapted from tests/test_seal_glm_observations.SealTests.fixture with the
        tail shard (s1) actually materialized and correctly identified — the stock
        fixture never loads uncommitted shards, this merger does."""
        from reap.glm53_prepare_record_tokens import record_identity, SOURCE_SHA256, TOKENIZER_SHA256
        token = base / 'tokens'
        token.mkdir()

        def shard_file(name, start, count, domain):
            rows = [{'input_ids': [1], 'original_tokens': 1, 'source_row': start + i,
                     'id': str(start + i), 'truncated': False, 'domain': domain}
                    for i in range(count)]
            path = token / name
            path.write_text(json.dumps({'schema': 'glm53-record-token-shard-v1', 'records': rows}))
            return {'shard_id': None, 'start': start, 'end': start + count, 'sequence_count': count,
                    'tokens': count, 'lengths': [1] * count, 'path': name,
                    'sha256': digest_file(path), 'source_rows': list(range(start, start + count)),
                    'record_ids': [str(start + i) for i in range(count)]}

        first = shard_file('s0.json', 0, 64, 'coding')
        first['shard_id'] = 0
        final = shard_file('s1.json', 64, 1, 'science')
        final['shard_id'] = 1
        manifest = {'schema': 'glm53-record-token-manifest-v1', 'state': 'COMPLETE',
                    'sequence_length': 16384, 'corpus': 'ours',
                    'identity': record_identity(SOURCE_SHA256, TOKENIZER_SHA256),
                    'sequence_count': 65, 'tokens': 65, 'shards': [first, final],
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
        receipt = make_receipt(contract, first, obs, runtime)
        sp = root / 'shard-00000'
        sp.mkdir()
        (sp / 'receipt.json').write_text(json.dumps(receipt))
        (root / 'record-status.json').write_text(json.dumps({'state': 'PAUSED_AT_RECORD_SHARD',
            'run_id': contract['run_id'], 'completed_shards': [0], 'records': 64, 'tokens': 64}))
        output = base / 'sealed'
        run_seal(type('NS', (), {'token_manifest': mp, 'model_identity': modelp,
                                 'run_root': root, 'output': output, 'keep': [8],
                                 'seed': 42}))
        return {'token_manifest': mp, 'model_identity': modelp, 'output': output}

    def merged_fixture(self, base, *, completion_runtime=None, tail_ids=None, tamper=False):
        partial = self.partial_fixture(base)
        mp = partial['token_manifest']
        manifest = json.loads(mp.read_text())
        contract = record_contract(json.loads(partial['model_identity'].read_text()),
                                   manifest, digest_file(mp))
        # Completion run-root for the tail shard, under a distinct output root.
        completion_root = base / 'completion-output' / contract['run_id']
        completion_root.mkdir(parents=True)
        shard1 = {k: manifest['shards'][1][k] for k in
                  ('shard_id', 'start', 'end', 'sequence_count', 'tokens', 'lengths',
                   'path', 'sha256')}
        runtime = completion_runtime or {'prefill_policy': 'completion', 'prefill_chunk_size': 512,
                                         'kda_implementation': {'module': 'test2', 'function': 'test2',
                                                                'package_version': '2'},
                                         'expert_implementation': {'backend': 'portable',
                                                                   'adapter_sha256': 'c' * 64}}
        row = {'expert_frequency': [1] * 8 + [0] * 280, 'total_tokens': 1}
        for key in ('ean_sum', 'weighted_ean_sum', 'weighted_ean_model_scaled_sum',
                    'weighted_expert_frequency_sum', 'max_activations'):
            row[key] = [2.] * 8 + [0.] * 280
        obs = {str(i): row for i in range(3, 45)}
        ids = [1] if tail_ids is None else tail_ids
        for sid in ids:
            receipt = make_receipt(contract, shard1, obs, runtime)
            sp = completion_root / f'shard-{sid:05d}'
            sp.mkdir()
            (sp / 'receipt.json').write_text(json.dumps(receipt))
        (completion_root / 'record-status.json').write_text(json.dumps(
            {'state': 'RECORD_LOCAL_COMPLETE', 'run_id': contract['run_id'],
             'completed_shards': ids, 'records': len(ids), 'tokens': len(ids)}))
        driver = base / 'derived-driver.py'
        driver.write_text('# derived capture driver\n')
        merged = base / 'sealed-complete'
        if tamper:
            path = partial['output'] / 'observations.json'
            path.write_text(path.read_text().replace('"records": 64', '"records": 63'))
        argv = ['--partial-observations', partial['output'] / 'observations.json',
                '--partial-seal', partial['output'] / 'seal.json',
                '--model-identity', partial['model_identity'],
                '--token-manifest', mp,
                '--completion-run-root', base / 'completion-output',
                '--completion-driver', driver,
                '--output', merged, '--keep', '4']
        return argv, merged, partial

    def test_complete_merge_matches_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            argv, merged, _ = self.merged_fixture(base)
            result = run_merge(argv)
            self.assertEqual(result['state'], 'SEALED_COMPLETE_ORIGINAL_RECORDS')
            self.assertEqual(result['records'], 65)
            self.assertEqual(result['completion_records'], 1)
            observations = json.loads((merged / 'observations.json').read_text())
            self.assertEqual(observations['covered_domain_records'], {'coding': 64, 'science': 1})
            self.assertEqual(observations['covered_domain_tokens'], {'coding': 64, 'science': 1})
            self.assertEqual([s['shard_id'] for s in observations['sources']], [0, 1])
            self.assertEqual([s['runtime_epoch'] for s in observations['sources']],
                             ['original', 'completion'])
            self.assertEqual(observations['missing_domains'], [])
            self.assertEqual(observations['committed_shards'], [0, 1])
            self.assertEqual(observations['observations']['3']['total_tokens'], 65)
            self.assertEqual(observations['observations']['3']['ean_sum'], [3.] * 8 + [0.] * 280)
            self.assertEqual(observations['domain_observations']['science']['3']['ean_sum'],
                             [2.] * 8 + [0.] * 280)
            self.assertEqual(observations['domain_observations']['coding']['3']['ean_sum'],
                             [1.] * 8 + [0.] * 280)
            self.assertEqual(observations['minimum_expert_routes'], 0)
            self.assertEqual(observations['runtime_identity']['prefill_policy'], 'test')
            self.assertEqual(observations['completion_runtime_identity']['prefill_policy'],
                             'completion')
            self.assertEqual(observations['completion_run']['phase'],
                             'tail shards 1-1 only (original run paused at shard 0; capture host pair changed)')
            summary = json.loads((merged / 'seal.json').read_text())
            self.assertEqual(summary['artifact_sha256']['observations.json'],
                             digest_file(merged / 'observations.json'))
            self.assertIn('merge_sealer_sha256', summary)
            candidates = json.loads((merged / 'candidates.json').read_text())
            self.assertEqual(candidates['source_seal_state'], 'SEALED_COMPLETE_ORIGINAL_RECORDS')
            self.assertIn('4', candidates['selection']['mass_global']['3']['keep_maps'])

    def test_refuses_paused_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            argv, merged, _ = self.merged_fixture(base)
            path = base / 'completion-output' / self.only_dir(base / 'completion-output') / 'record-status.json'
            value = json.loads(path.read_text())
            value['state'] = 'PAUSED_AT_RECORD_SHARD'
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                run_merge(argv)
            self.assertFalse(merged.exists())

    def test_refuses_incomplete_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            argv, merged, _ = self.merged_fixture(base, tail_ids=[])
            with self.assertRaises(ValueError):
                run_merge(argv)
            self.assertFalse(merged.exists())

    def test_refuses_runtime_split_within_completion_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            argv, merged, _ = self.merged_fixture(base)
            path = base / 'completion-output' / self.only_dir(base / 'completion-output') / 'shard-00001' / 'receipt.json'
            receipt = json.loads(path.read_text())
            receipt['runtime_identity']['prefill_policy'] = 'tampered'
            path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                run_merge(argv)
            self.assertFalse(merged.exists())

    def test_refuses_tampered_partial_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            argv, merged, _ = self.merged_fixture(base, tamper=True)
            with self.assertRaises(ValueError):
                run_merge(argv)
            self.assertFalse(merged.exists())

    @staticmethod
    def only_dir(path):
        entries = [p for p in path.iterdir() if p.is_dir()]
        assert len(entries) == 1
        return entries[0].name


if __name__ == '__main__':
    unittest.main(verbosity=2)
