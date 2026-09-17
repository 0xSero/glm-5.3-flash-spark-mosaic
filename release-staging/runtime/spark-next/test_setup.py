"""Small CPU fixtures exercise corruption failures, not a fake model load."""
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import setup


class SetupTest(unittest.TestCase):
    def fixture(self, root, offset=2):
        header = json.dumps({'weight': {'dtype': 'BF16', 'shape': [1], 'data_offsets': [0, offset]}}).encode()
        p = root / 'one.safetensors'
        p.write_bytes(struct.pack('<Q', len(header)) + header + b'12')
        (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': p.name}}))
        (root / 'EXL3_MANIFEST.json').write_text('{}')
        return {'files': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': setup.digest(p)}], 'weight_file_count': 1, 'indexed_tensor_count': 1, 'metadata_sha256': {n: setup.digest(root / n) for n in ('model.safetensors.index.json', 'EXL3_MANIFEST.json')}}

    def test_valid_hash_and_tensor_closure(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            p = Path(td); lock = self.fixture(p)
            self.assertEqual(setup.verify(p, lock)['tensor_count'], 1)

    def test_corrupted_weight_fails(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); lock = self.fixture(p)
            f = p / 'one.safetensors'; f.write_bytes(f.read_bytes()[:-1] + b'x')
            with self.assertRaisesRegex(ValueError, 'Weight size/hash'):
                setup.verify(p, lock)

    def test_corrupted_metadata_fails(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); lock = self.fixture(p); (p / 'EXL3_MANIFEST.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Metadata hash'):
                setup.verify(p, lock)

    def test_extra_weights_fail(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); lock = self.fixture(p); (p / 'extra.safetensors').write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'Weight file set'):
                setup.verify(p, lock)

    def test_payload_offsets_must_close(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); lock = self.fixture(p, offset=3)
            with self.assertRaisesRegex(ValueError, 'payload/file closure'):
                setup.verify(p, lock)

    def test_index_target_must_exist(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); lock = self.fixture(p); idx = p / 'model.safetensors.index.json'
            idx.write_text(json.dumps({'weight_map': {'weight': 'absent.safetensors'}}))
            lock['metadata_sha256'][idx.name] = setup.digest(idx)
            with self.assertRaisesRegex(ValueError, 'weight/index closure'):
                setup.verify(p, lock)

    def test_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); (p / 'escape').symlink_to('/etc/passwd')
            for name in ('../elsewhere', '/etc/passwd', 'escape'):
                with self.assertRaises(ValueError):
                    setup.contained(p, name)

    def test_default_runtime_has_explicit_required_features(self):
        with patch.dict(os.environ, {}, clear=True):
            args, config = setup.runtime_args()
        self.assertEqual(args[args.index('--kv-cache-dtype') + 1], 'fp8_ds_mla')
        self.assertEqual(args[args.index('--attention-backend') + 1], 'B12X')
        self.assertEqual(args[:5], [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', '/model'])
        self.assertEqual(args[args.index('--block-size') + 1], '256')
        self.assertEqual(json.loads(args[args.index('--additional-config') + 1]), {'kda_prefill_backend': 'b12x'})
        self.assertEqual(config['speculative_config'], {'method': 'mtp', 'model': '/state/mtp', 'num_speculative_tokens': 1, 'attention_backend': 'B12X'})
        self.assertEqual(config['compilation_config']['cudagraph_capture_sizes'], [2])
        self.assertEqual(config['max_model_len'], 262144)
        self.assertNotIn('--profiler-config', args)

    def test_public_model_lock_is_complete(self):
        self.assertEqual(len(setup.LOCK['files']), 133)
        self.assertEqual(sum(r['bytes'] for r in setup.LOCK['files']), 111352026456)
        self.assertEqual(len(setup.LOCK['revision']), 40)
        self.assertIn('processor_config.json', setup.LOCK['metadata_sha256'])
        self.assertIn('tokenizer.json', setup.LOCK['metadata_sha256'])

    def test_metrics_waits_for_delayed_export(self):
        names = ('vllm:spec_decode_num_draft_tokens_total', 'vllm:spec_decode_num_accepted_tokens_total')
        before = dict(zip(names, (0, 0))); after = dict(zip(names, (4, 3)))
        clock = [0.0]
        with patch.object(setup, 'counters', side_effect=[before, before, after]), patch.object(setup.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(setup.time, 'sleep', side_effect=lambda n: clock.__setitem__(0, clock[0] + n)):
            self.assertEqual(setup.wait_mtp_counters(before), after)
        self.assertEqual(clock[0], 2)

    def test_metrics_rejects_invalid_deltas_at_bounded_deadline(self):
        names = ('vllm:spec_decode_num_draft_tokens_total', 'vllm:spec_decode_num_accepted_tokens_total')
        before = dict(zip(names, (0, 0)))
        for values in ((0, 0), (2, 3), (float('inf'), 1), (3, float('nan')), (3, -1)):
            with self.subTest(values=values):
                clock = [0.0]
                with patch.object(setup, 'counters', return_value=dict(zip(names, values))), patch.object(setup.time, 'monotonic', side_effect=lambda: clock[0]), patch.object(setup.time, 'sleep', side_effect=lambda n: clock.__setitem__(0, clock[0] + n)):
                    with self.assertRaises(TimeoutError):
                        setup.wait_mtp_counters(before)
                self.assertEqual(clock[0], 15)

    def test_occupied_port_is_rejected_without_http(self):
        with socket.socket() as owner, patch.object(setup, 'http') as http:
            owner.bind(('127.0.0.1', 0)); owner.listen()
            with self.assertRaises(OSError):
                setup.check_bind_port('127.0.0.1', owner.getsockname()[1])
            http.assert_not_called()

    def test_generated_mtp_view_edits_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            view = Path(td); config = view / 'config.json'; config.write_text('{"precision":"native"}')
            link = view / 'weights'; link.symlink_to('/model/original.safetensors')
            expected = setup.view_identity(view); setup.verify_view(view, expected)
            config.write_text('{"precision":"changed"}')
            with self.assertRaises(RuntimeError): setup.verify_view(view, expected)
            config.write_text('{"precision":"native"}')
            link.unlink(); link.symlink_to('/model/other.safetensors')
            with self.assertRaises(RuntimeError): setup.verify_view(view, expected)

    def test_startup_clears_stale_ready_even_if_gpu_guard_fails(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td); ready = state / 'fresh-text.json'; ready.write_text('{}')
            with patch.object(setup, 'STATE', state), patch.object(setup, 'gpu_check', side_effect=RuntimeError('busy')):
                with self.assertRaisesRegex(RuntimeError, 'busy'): setup.serve()
            self.assertFalse(ready.exists())

    def shell_trace(self, root, busy=False):
        binary = root / 'docker'
        binary.write_text(f'#!{sys.executable}\nimport json,os,sys\na=sys.argv[1:]\nwith open(os.environ["TRACE"],"a") as f:f.write(json.dumps(a)+"\\n")\nif a[:2]==["container","inspect"]:sys.exit(1)\nif a[-1]=="gpu-check" and os.getenv("BUSY")=="1":sys.exit(3)\n')
        binary.chmod(0o755); trace = root / 'trace.jsonl'
        env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'], TRACE=str(trace), GLM53_IMAGE='sha256:'+'a'*64, BUSY='1' if busy else '0')
        result = subprocess.run(['bash', str(Path(setup.__file__).parent / 'run.sh'), str(root / 'model'), str(root / 'state')], env=env, capture_output=True, text=True)
        return result, [json.loads(line) for line in trace.read_text().splitlines()]

    def test_shell_guards_bracket_capped_setup(self):
        with tempfile.TemporaryDirectory() as td:
            result, trace = self.shell_trace(Path(td)); self.assertEqual(result.returncode, 0, result.stderr)
            guards = [i for i, args in enumerate(trace) if args[-1] == 'gpu-check']
            prepare = next(i for i, args in enumerate(trace) if args[-1] == 'prepare')
            self.assertLess(guards[0], prepare); self.assertGreater(guards[1], prepare)
            args = trace[prepare]
            for flag, value in (('--cpus', '2'), ('--memory', '3g'), ('--memory-swap', '3g')):
                self.assertEqual(args[args.index(flag)+1], value)
            self.assertNotIn('--gpus', args)
            serve_args = next(args for args in trace if args[-1] == 'serve')
            for flag in ('VLLM_MXFP8_LM_HEAD=0', 'VLLM_MTP_NVFP4_LM_HEAD=0'):
                self.assertIn(flag, serve_args)
                self.assertEqual(serve_args[serve_args.index(flag)-1], '-e')

    def test_busy_gpu_prevents_heavy_setup(self):
        with tempfile.TemporaryDirectory() as td:
            result, trace = self.shell_trace(Path(td), busy=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(any(args[-1] == 'prepare' for args in trace))


if __name__ == '__main__':
    unittest.main()
