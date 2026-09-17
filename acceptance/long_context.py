#!/usr/bin/env python3
"""Near-limit semantic retrieval with explicit output reserve and server token count."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets

from run import FILLER, chat_tokens, metrics, request_json, sha, stream, write, validate_runtime
from vision_acceptance import score


def prepare_retrieval(tokenizer, target, template):
    expected={name:secrets.token_hex(8) for name in ('amber','birch','cobalt','dahlia')}
    text='Independent retrieval trial '+secrets.token_hex(16)+'.\n'
    filler=tokenizer.encode(FILLER,add_special_tokens=False)
    if not filler:raise ValueError('empty filler')
    def pad(n):return tokenizer.decode((filler*((n+len(filler)-1)//len(filler)))[:n],skip_special_tokens=False)
    positions={}
    for fraction,(name,value) in zip((.05,.35,.65,.95),expected.items()):
        text+=pad(max(0,int(target*fraction)-chat_tokens(tokenizer,text,template)))
        positions[name]=chat_tokens(tokenizer,text,template)
        text+=f'\nRegistry record: {name} has retrieval value {value}.\n'
    question='\nReturn only a JSON object with keys amber, birch, cobalt, dahlia and their exact registry values.\n'
    budget=max(0,target-chat_tokens(tokenizer,text+question,template))
    for _ in range(12):
        prompt=text+pad(budget)+question
        count=chat_tokens(tokenizer,prompt,template)
        if target-4<=count<=target:return prompt,expected,positions,count
        budget=max(0,budget+target-count)
    raise ValueError('failed to fit retrieval prompt')


def evaluate(row,expected,context,reserve):
    correct,actual=score(row.get('content',''),expected)
    usage=row.get('usage') or {};prompt=usage.get('prompt_tokens');completion=usage.get('completion_tokens')
    count_ok=(type(prompt) is int and type(completion) is int and
              context-reserve-4<=prompt<=context-reserve and prompt+completion<=context)
    return {'passed':bool(row.get('success') and row.get('finish_reason')=='stop' and correct and count_ok),
            'semantic_retrieval_pass':correct,'context_accounting_pass':count_ok,'actual':actual,
            'context_limit':context,'output_reserved_tokens':reserve,
            'actual_prompt_tokens':prompt,'actual_completion_tokens':completion,
            'scope':'Four separated synthetic records; near-limit capacity and retrieval, not broad long-context reasoning.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True);p.add_argument('--metrics-url',required=True)
    p.add_argument('--model',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--tokenizer-revision')
    p.add_argument('--runtime-receipt',type=Path,required=True)
    p.add_argument('--context',type=int,default=262144);p.add_argument('--output-reserve',type=int,default=2048)
    p.add_argument('--template-json',default='{"enable_thinking":true}')
    p.add_argument('--timeout',type=int,default=7200);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.context!=262144 or not 128<=a.output_reserve<=4096:p.error('requires262144context and128..4096output reserve')
    if a.output.exists():p.error('use a fresh output directory')
    validate_runtime(json.loads(a.runtime_receipt.read_text()),a.model,a.context)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer,revision=a.tokenizer_revision,trust_remote_code=False)
    template=json.loads(a.template_json)
    prompt,expected,positions,count=prepare_retrieval(tok,a.context-a.output_reserve,template)
    a.output.mkdir(parents=True)
    receipt={'model':a.model,'expected':expected,'positions':positions,'local_chat_tokens':count,
             'tokenizer_revision':a.tokenizer_revision,'template':template,'runtime_receipt_sha256':sha(a.runtime_receipt),
             'source_sha256':sha(__file__),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
    (a.output/'prompt.txt').write_text(prompt);write(a.output/'prepared.json',receipt)
    before=metrics(a.metrics_url,a.model)
    if not before['idle']:raise ValueError('server not idle')
    actual=request_json(a.base_url.rstrip('/').removesuffix('/v1')+'/tokenize',
           {'model':a.model,'messages':[{'role':'user','content':prompt}],
            'add_generation_prompt':True,'chat_template_kwargs':template})['count']
    if actual!=count or actual+a.output_reserve>a.context:raise ValueError('server tokenization mismatch')
    response=stream(a.base_url,a.model,prompt,template,a.output_reserve,a.timeout)
    result=evaluate(response,expected,a.context,a.output_reserve)
    cached=((response.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens')
    result['cached_tokens']=cached
    if cached not in (None,0):result.update(passed=False,cold_cache_pass=False)
    result['idle_after']=metrics(a.metrics_url,a.model)['idle']
    result['passed']=result['passed'] and result['idle_after']
    write(a.output/'result.json',{'prepared':receipt,'evaluation':result,'response':response})
    print(json.dumps(result))
    raise SystemExit(not result['passed'])


if __name__=='__main__':main()
