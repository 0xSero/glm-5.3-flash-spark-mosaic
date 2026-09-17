import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('parity', Path(__file__).parents[1] / 'runtime_parity.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class ParityTest(unittest.TestCase):
    def test_position(self):
        prefix, label = p.position_contract(list(range(2048)))
        self.assertEqual(len(prefix), 2047)
        self.assertEqual(prefix[-1], 2046)
        self.assertEqual(label, 2047)
        with self.assertRaises(ValueError):
            p.position_contract(list(range(2048)), 2047)

    def test_terminal(self):
        data = {'Id': 'capture', 'State': {'Running': False, 'ExitCode': 0, 'FinishedAt': '2026-09-11T12:00:00Z'}}
        self.assertEqual(p.terminal_contract([data]), 'capture')
        data['State']['Running'] = True
        with self.assertRaises(ValueError):
            p.terminal_contract(data)

    def test_compare_and_unsupported(self):
        probe = {'row': 0, 'prefix_ids': [1, 2], 'expected_top1': 3, 'top1_gap': .1,
                 'expected_top_logprobs': {'3': -.4, '4': -.5}}
        response = {'usage': {'prompt_tokens': 2, 'completion_tokens': 1}, 'choices': [
            {'logprobs': {'tokens': ['token_id:3'], 'top_logprobs': [{'token_id:3': -.41, 'token_id:4': -.52}]}}]}
        self.assertTrue(p.compare_response(probe, response, .05)['strict_probe_pass'])
        self.assertFalse(p.compare_response(probe, response, .001)['strict_probe_pass'])
        response['choices'][0]['logprobs']['tokens'] = ['hello']
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED'):
            p.compare_response(probe, response, .05)

    def test_token_accounting(self):
        with self.assertRaisesRegex(ValueError, 'accounting'):
            p.compare_response({'prefix_ids': [1]}, {'usage': {'prompt_tokens': 2}}, .05)


if __name__ == '__main__':
    unittest.main()
