import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT))
from timing import capacity_status, matched_window, summarize
from run import render, validate_runtime
from long_context import evaluate
from mtp_proof import prove
from vision_acceptance import payload, score


def row(times,counts):
    return {'success':True,'started_monotonic':-.5,'usage':{'prompt_tokens':100,'completion_tokens':sum(counts)},
            'events':[{'monotonic':t,'token_ids':list(range(n))} for t,n in zip(times,counts)]}


class AcceptanceTests(unittest.TestCase):
    def test_common_window_total_not_sum_of_differently_timed_rates(self):
        a=row([0,2,4,6],[2,2,2,2]);b=row([1,3,5,7],[3,3,3,3])
        result=matched_window([a,b],minimum_seconds=4)
        self.assertEqual(result['status'],'MATCHED_SUSTAINED')
        self.assertEqual(result['window_seconds'],5)
        self.assertEqual(result['total_aggregate_decode_tok_s'],2.4)
        self.assertEqual(result['mean_per_request_decode_tok_s'],1.2)
        self.assertEqual([s['tokens'] for s in result['streams']],[6,6])

    def test_no_guessing_token_ids_or_silent_windows(self):
        a=row([0,10],[2,3])
        self.assertEqual(matched_window([a])['status'],'UNAVAILABLE_SILENT_STREAM')
        a['events'][0]['token_ids']=None
        self.assertEqual(matched_window([a])['status'],'UNAVAILABLE_EXACT_TOKEN_IDS')
        a=row([0,1],[2,3]);a['usage']['completion_tokens']=10
        self.assertEqual(matched_window([a])['status'],'UNAVAILABLE_TOKEN_ACCOUNTING_MISMATCH')

    def test_short_window_remains_screen_and_failed_cell_has_no_total(self):
        a=row([0,1],[2,3])
        self.assertEqual(matched_window([a])['status'],'MATCHED_SCREEN')
        a['success']=False
        r=summarize([a],3)
        self.assertNotIn('total_output_tokens_per_end_to_end_wall_second',r)

    def test_capacity_counts_output_reserve_and_active_slots(self):
        self.assertEqual(capacity_status(260096,2048,262144,1,1,262144),'PLANNED')
        self.assertEqual(capacity_status(262144,2048,262144,1,1,262144),'UNSUPPORTED_REQUEST_CONTEXT')
        self.assertEqual(capacity_status(260096,2048,262144,2,1,524288),'UNSUPPORTED_ACTIVE_SLOTS')
        self.assertEqual(capacity_status(260096,2048,262144,2,2,400000),'UNSUPPORTED_KV_CAPACITY')

    def test_nearmax_semantics_and_context_accounting(self):
        a={'success':True,'finish_reason':'stop','content':'{"x":"123"}',
           'usage':{'prompt_tokens':260096,'completion_tokens':20}}
        self.assertTrue(evaluate(a,{'x':'123'},262144,2048)['passed'])
        a['usage']['prompt_tokens']=262144
        self.assertFalse(evaluate(a,{'x':'123'},262144,2048)['passed'])
        a['usage']['prompt_tokens']=260096;a['content']='{"x":"wrong"}'
        self.assertFalse(evaluate(a,{'x':'123'},262144,2048)['passed'])

    def test_mtp_config_is_not_execution_proof(self):
        r={'speculative_config':{'method':'mtp','num_speculative_tokens':1}}
        c={'status':'COMPLETED','speculative_counter_deltas':{},'native_timing':{'status':'matched_isolated_cell'}}
        self.assertFalse(prove(r,c)['passed'])
        c['speculative_counter_deltas']={'vllm:spec_decode_num_draft_tokens_total':100,'vllm:spec_decode_num_accepted_tokens_total':70}
        self.assertTrue(prove(r,c)['passed'])
        self.assertEqual(prove(r,c)['accepted_fraction'],.7)
        r['speculative_config']['method']='dflash'
        self.assertFalse(prove(r,c)['passed'])

    def test_runtime_receipt_capacity_must_match(self):
        r={'model':'x','context_limit':262144,'max_num_seqs':1,'kv_capacity_tokens':300000,
           'prefix_caching':False,'cuda_graphs':True,'evidence_sha256':{'engine_log':'a'*64}}
        validate_runtime(r,'x',262144,1,300000)
        with self.assertRaises(ValueError):validate_runtime(r,'x',262144,8,300000)

    def test_visual_fixtures_integrity_and_no_expected_object_in_request(self):
        root=ROOT/'vision-fixtures';cases=json.loads((root/'cases.json').read_text())['cases']
        for case in cases:
            request=payload(case,root,'test')
            self.assertNotIn('expected',request)
            self.assertTrue(score(json.dumps(case['expected']),case['expected'])[0])
        self.assertEqual({c['kind'] for c in cases},{'image','video'})

    def test_plan_cli_does_not_need_tokenizer_or_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'plan'
            subprocess.run([sys.executable,str(ROOT/'run.py'),'--plan-only','--model','test',
                '--max-num-seqs','1','--kv-capacity-tokens','262144','--repeats','1','--output',str(out)],check=True)
            cells=json.loads((out/'matrix.json').read_text())
            self.assertEqual(len(cells),28)
            self.assertEqual(cells[-4]['input_tokens'],260096)
            self.assertEqual(cells[-1]['status'],'UNSUPPORTED_ACTIVE_SLOTS')
            table=(out/'TABLE.md').read_text()
            self.assertIn('TOTAL matched decode tok/s',table)
            self.assertNotIn('0.00',table)


if __name__=='__main__':unittest.main()
