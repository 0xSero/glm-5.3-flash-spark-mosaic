import copy
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from serving_quality import score


class ServingQualityTest(unittest.TestCase):
    def fixture(self):
        ids=[198]+[44]*2047
        alternatives={'token_id:44':-.7, **{f'token_id:{i}':-5-i*.01 for i in range(1,20)}}
        response={'usage':{'prompt_tokens':2048,'completion_tokens':1},'choices':[{'logprobs':{
            'tokens':['token_id:198']+['token_id:44']*2048,
            'token_logprobs':[None]+[-.7]*2048,
            'top_logprobs':[None]+[alternatives.copy() for _ in range(2048)]}}]}
        return ids,[44]*2047,response

    def test_exact_scored_positions_exclude_generated_token(self):
        ids,teacher,response=self.fixture()
        response['choices'][0]['logprobs']['token_logprobs'][-1]=-99
        measured=score(ids,teacher,response)
        self.assertEqual(measured['positions'],2047)
        self.assertAlmostEqual(measured['runtime_nll_sum'],2047*.7)
        self.assertEqual(measured['teacher_top1_agree'],2047)

    def test_changed_or_incomplete_echo_fails(self):
        for kind in ('ids','missing','nan','ground_truth'):
            ids,teacher,response=self.fixture();logs=response['choices'][0]['logprobs']
            if kind=='ids':logs['tokens'][10]='token_id:45'
            if kind=='missing':logs['token_logprobs'].pop()
            if kind=='nan':logs['token_logprobs'][10]=math.nan
            if kind=='ground_truth':logs['top_logprobs'][10]['token_id:44']=-.8
            with self.assertRaises(ValueError,msg=kind):score(ids,teacher,response)

    def test_uncovered_ties_never_inflate_agreement(self):
        ids,teacher,response=self.fixture();logs=response['choices'][0]['logprobs']
        logs['top_logprobs'][1]={k:-.7 for k in logs['top_logprobs'][1]}
        result=score(ids,teacher,response)
        self.assertEqual(result['argmax_covered_positions'],2046)
        self.assertEqual(result['teacher_top1_agree'],2046)
        self.assertEqual(result['tied_max_positions'],1)


if __name__=='__main__':unittest.main()
