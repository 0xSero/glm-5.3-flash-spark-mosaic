#!/usr/bin/env python3
"""Reproducible isolated streaming matrix; use --plan-only before any load."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid

import native_metrics
from timing import capacity_status, summarize

FILLER = 'The archive records a cache entry. A reader verifies the revision before reuse. Workers preserve unrelated records.\n'
TASK = '\nWrite 240 numbered Python comment lines with distinct, concrete debugging tips. Finish after line 240.\n'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def request_json(url, body=None, timeout=30):
    headers={'Content-Type':'application/json'}
    key=os.environ.get('MODEL_API_KEY')
    if key:headers['Authorization']='Bearer '+key
    req=urllib.request.Request(url,data=None if body is None else json.dumps(body).encode(),headers=headers)
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.load(response)


def metrics(url, model):
    headers={}
    if os.environ.get('MODEL_API_KEY'):headers['Authorization']='Bearer '+os.environ['MODEL_API_KEY']
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=15) as r:
        data=r.read(8*1024*1024+1)
    if len(data)>8*1024*1024:raise ValueError('oversized metrics response')
    text=data.decode();idle={};mtp={}
    for line in text.splitlines():
        match=native_metrics.LINE.fullmatch(line)
        if not match:continue
        label=native_metrics.MODEL.search(match[2] or '')
        if label and json.loads(label[1])!=model:continue
        if match[1] in ('vllm:num_requests_running','vllm:num_requests_waiting'):
            idle[match[1]]=idle.get(match[1],0)+float(match[3])
        if match[1].startswith('vllm:spec_decode_') and match[1].endswith('_total'):
            mtp[match[1]]=mtp.get(match[1],0)+float(match[3])
    if len(idle)!=2:raise ValueError('idle gauges unavailable; exclusivity unproven')
    if any(not math.isfinite(v) or v<0 for v in list(idle.values())+list(mtp.values())):
        raise ValueError('invalid server gauges or speculative counters')
    try:timing=native_metrics.parse(text,model)
    except ValueError:timing=None
    return {'idle':all(v==0 for v in idle.values()),'gauges':idle,'request_metrics':timing,'speculative_counters':mtp}


def validate_runtime(receipt, model, context, slots=None, kv=None):
    if receipt.get('model')!=model or receipt.get('context_limit')!=context:
        raise ValueError('runtime receipt model/context mismatch')
    if receipt.get('prefix_caching') is not False or receipt.get('cuda_graphs') is not True:
        raise ValueError('cold-cache and CUDA-graph runtime evidence required')
    if slots is not None and receipt.get('max_num_seqs')!=slots:
        raise ValueError('runtime active-slot receipt mismatch')
    if kv is not None and receipt.get('kv_capacity_tokens')!=kv:
        raise ValueError('runtime KV capacity receipt mismatch')
    evidence=receipt.get('evidence_sha256') or {}
    if not evidence or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in evidence.values()):
        raise ValueError('runtime receipt must bind original evidence hashes')


def chat_tokens(tokenizer, prompt, template):
    return len(tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,
                 return_dict=False,add_generation_prompt=True,**template))


def prepare(tokenizer, target, nonce, template):
    prefix='Independent run '+nonce+'. Treat this archive as background context.\n'
    ids=tokenizer.encode(FILLER,add_special_tokens=False)
    if not ids:raise ValueError('empty filler')
    budget=max(0,target-chat_tokens(tokenizer,prefix+TASK,template))
    for _ in range(12):
        text=prefix+tokenizer.decode((ids*((budget+len(ids)-1)//len(ids)))[:budget],skip_special_tokens=False)+TASK
        count=chat_tokens(tokenizer,text,template)
        if target-4<=count<=target:return text,count
        budget=max(0,budget+target-count)
    raise ValueError('could not form prompt within four tokens of target')


def stream(base, model, prompt, template, reserve, timeout, barrier=None):
    payload={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,
             'max_tokens':reserve,'stream':True,'stream_options':{'include_usage':True},
             'return_token_ids':True,'chat_template_kwargs':template}
    headers={'Content-Type':'application/json'}
    if os.environ.get('MODEL_API_KEY'):headers['Authorization']='Bearer '+os.environ['MODEL_API_KEY']
    if barrier:barrier.wait()
    started=time.monotonic();events=[];content=[];reasoning=[];usage=None;finish=None;done=False
    row={'success':False,'started_monotonic':started,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
    try:
        req=urllib.request.Request(base.rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers=headers)
        with urllib.request.urlopen(req,timeout=timeout) as response:
            for raw in response:
                now=time.monotonic()
                if now-started>timeout:raise TimeoutError('wall deadline')
                line=raw.decode().strip()
                if not line.startswith('data:'):continue
                value=line[5:].strip()
                if value=='[DONE]':done=True;break
                chunk=json.loads(value)
                if chunk.get('error'):raise ValueError('server stream error')
                if chunk.get('usage'):usage=chunk['usage']
                for choice in chunk.get('choices',[]):
                    if choice.get('index',0)!=0:continue
                    delta=choice.get('delta') or {};ids=choice.get('token_ids')
                    if ids is not None and (not isinstance(ids,list) or any(type(i) is not int for i in ids)):
                        raise ValueError('malformed emitted token IDs')
                    text=delta.get('content') or '';reason=delta.get('reasoning_content') or delta.get('reasoning') or ''
                    content.append(text);reasoning.append(reason)
                    if text or reason or ids:
                        events.append({'monotonic':now,'token_ids':ids,'content':text,'reasoning':reason})
                    if choice.get('finish_reason') is not None:finish=choice['finish_reason']
        if not done or finish not in ('stop','length') or not events or not usage:
            raise ValueError('incomplete response')
        if any(type(usage.get(k)) is not int or usage[k]<=0 for k in ('completion_tokens','prompt_tokens')):
            raise ValueError('missing token usage')
        row['success']=True
    except Exception as e:
        row.update(error_type=type(e).__name__)
        if isinstance(e,urllib.error.HTTPError):
            row['http_status']=e.code
            row['error_body']=e.read(2048).decode(errors='replace')
    row.update(events=events,usage=usage,finish_reason=finish,done=done,
               content=''.join(content),reasoning=''.join(reasoning),ended_monotonic=time.monotonic())
    return row


def render(cells):
    lines=['# Measured inference matrix','','No inferred or estimated value is placed in the matched decode columns.',
           'Request-native prefill uses summed overlapping request durations; it is not aggregate GPU throughput.','',
           '| Prompt target | Actual prompt/request | C | Repeat | Status | TOTAL matched decode tok/s | Mean/request matched decode tok/s | Window s | Server request-prefill tok/s | Prompt/TTFT effective tok/s | TTFT p50/p90 s | Client decode estimate tok/s | End-to-end total tok/s |',
           '|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|']
    def f(v):return '—' if v is None else f'{v:.2f}'
    for c in cells:
        s=c.get('summary',{});m=s.get('matched_decode',{}) if c['status']=='COMPLETED' else {};n=c.get('native_timing',{}) if c['status']=='COMPLETED' else {}
        status=c['status']
        if status=='COMPLETED':status=m.get('status','UNAVAILABLE')
        actual=', '.join(map(str,c.get('actual_prompt_tokens',[]))) or '—'
        lines.append('| '+' | '.join([str(c['input_tokens']),actual,str(c['concurrency']),str(c['repeat']),status,
            f(m.get('total_aggregate_decode_tok_s')),f(m.get('mean_per_request_decode_tok_s')),f(m.get('window_seconds')),
            f(n.get('prompt_tokens_per_sum_request_prefill_second')),f(s.get('mean_prompt_tokens_per_ttft')),
            f(s.get('ttft_p50_seconds'))+'/'+f(s.get('ttft_p90_seconds')),f(s.get('mean_client_decode_tok_s_estimate')),
            f(s.get('total_output_tokens_per_end_to_end_wall_second'))])+' |')
    return '\n'.join(lines)+'\n'


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',help='OpenAI base ending /v1; not saved')
    p.add_argument('--metrics-url',help='Required to prove idle and match native timing; not saved')
    p.add_argument('--model',required=True);p.add_argument('--tokenizer');p.add_argument('--tokenizer-revision')
    p.add_argument('--runtime-receipt',type=Path,help='Pinned actual runtime/config evidence, required for execution')
    p.add_argument('--context',type=int,default=262144);p.add_argument('--output-reserve',type=int,default=2048)
    p.add_argument('--max-num-seqs',type=int,required=True);p.add_argument('--kv-capacity-tokens',type=int,required=True)
    p.add_argument('--lengths',default='1024,4096,16384,65536,131072,200000,nearmax')
    p.add_argument('--concurrencies',default='1,2,4,8');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--timeout',type=int,default=7200);p.add_argument('--minimum-window',type=float,default=30)
    p.add_argument('--maximum-silence',type=float,default=5)
    p.add_argument('--template-json',default='{"enable_thinking":true}')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-only',action='store_true')
    return p


def main():
    p=parser();a=p.parse_args()
    if min(a.context,a.output_reserve,a.max_num_seqs,a.kv_capacity_tokens,a.repeats,a.timeout)<=0 or a.output_reserve>=a.context:
        p.error('positive sizes required; output reserve must be smaller than context')
    lengths=[a.context-a.output_reserve if x=='nearmax' else int(x) for x in a.lengths.split(',')]
    cs=sorted(set(int(x) for x in a.concurrencies.split(',')))
    if not cs or cs[0]!=1 or min(lengths)<=0:p.error('C1 first required; lengths positive')
    template=json.loads(a.template_json)
    a.output.mkdir(parents=True,exist_ok=False)
    cells=[{'input_tokens':l,'concurrency':c,'repeat':r,
            'status':capacity_status(l,a.output_reserve,a.context,c,a.max_num_seqs,a.kv_capacity_tokens)}
           for l in lengths for c in cs for r in range(a.repeats)]
    run={'schema':'glm53-reap-inference-matrix-v1','model':a.model,'context_limit':a.context,
         'output_reserve':a.output_reserve,'max_num_seqs':a.max_num_seqs,'kv_capacity_tokens':a.kv_capacity_tokens,
         'content_class':'structured','cache':'cold_requested','sampling':{'temperature':0},'template':template,
         'phase':'plan' if a.plan_only else 'screen_and_sustained','run_id':uuid.uuid4().hex,
         'tokenizer_revision':a.tokenizer_revision,'source_sha256':{n:sha(Path(__file__).with_name(n)) for n in ['run.py','timing.py','native_metrics.py']},
         'native_mtp_claim':False,'runtime_receipt_sha256':sha(a.runtime_receipt) if a.runtime_receipt else None}
    write(a.output/'run.json',run);write(a.output/'matrix.json',cells);(a.output/'TABLE.md').write_text(render(cells))
    if a.plan_only:return
    if not all((a.base_url,a.metrics_url,a.tokenizer,a.runtime_receipt)):
        p.error('execution requires base-url, metrics-url, tokenizer, runtime-receipt')
    validate_runtime(json.loads(a.runtime_receipt.read_text()),a.model,a.context,a.max_num_seqs,a.kv_capacity_tokens)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer,revision=a.tokenizer_revision,trust_remote_code=False)
    stopped=False
    for index,cell in enumerate(cells):
        if cell['status']!='PLANNED':continue
        if stopped:
            cell['status']='NOT_RUN_AFTER_FAILURE';continue
        try:
            before=metrics(a.metrics_url,a.model)
            if not before['idle']:raise ValueError('runtime not idle; other clients or queued work present')
            prepared=[prepare(tok,cell['input_tokens'],uuid.uuid4().hex,template) for _ in range(cell['concurrency'])]
            actual=[]
            for prompt,count in prepared:
                check=request_json(a.base_url.rstrip('/').removesuffix('/v1')+'/tokenize',
                    {'model':a.model,'messages':[{'role':'user','content':prompt}],
                     'add_generation_prompt':True,'chat_template_kwargs':template})
                actual.append(check['count'])
                if check['count']!=count:raise ValueError('server/local chat tokenization mismatch')
            if any(capacity_status(n,a.output_reserve,a.context,cell['concurrency'],a.max_num_seqs,a.kv_capacity_tokens)!='PLANNED' for n in actual):
                raise ValueError('actual server prompt exceeds declared context or KV capacity')
            barrier=threading.Barrier(cell['concurrency']);started=time.monotonic()
            with ThreadPoolExecutor(max_workers=cell['concurrency']) as pool:
                rows=list(pool.map(lambda item:stream(a.base_url,a.model,item[0],template,a.output_reserve,a.timeout,barrier),prepared))
            wall=time.monotonic()-started
            for row,n in zip(rows,actual):
                if row.get('success'):
                    usage=row['usage'];cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
                    row['cached_tokens']=cached
                    if usage['prompt_tokens']!=n or usage['prompt_tokens']+usage['completion_tokens']>a.context or cached not in (None,0):
                        row.update(success=False,error_type='TokenOrColdCacheContractMismatch')
            cell.update(actual_prompt_tokens=actual,status='COMPLETED' if all(r['success'] for r in rows) else 'FAILED_REQUEST',
                        summary=summarize(rows,wall,minimum_seconds=a.minimum_window,maximum_silence=a.maximum_silence))
            write(a.output/f'cell-{index:03d}.json',{'cell':cell,'requests':rows})
            after=metrics(a.metrics_url,a.model)
            cell['idle_after']=after['idle']
            if not after['idle']:cell['status']='FAILED_IDLE_AFTER'
            deadline=time.monotonic()+15
            while True:
                try:
                    cell['native_timing']=native_metrics.compare(before['request_metrics'],after['request_metrics'],rows)
                    break
                except (ValueError,TypeError,KeyError):
                    if before['request_metrics'] is None or not all(r['success'] for r in rows) or time.monotonic()>=deadline:
                        cell['native_timing']={'status':'UNAVAILABLE_OR_UNMATCHED'}
                        if before['request_metrics'] is not None and all(r['success'] for r in rows):
                            cell['status']='FAILED_NATIVE_ACCOUNTING'
                            cell['summary']['matched_decode']={'status':'UNAVAILABLE_UNMATCHED_ISOLATION'}
                        break
                    time.sleep(1)
                    after=metrics(a.metrics_url,a.model)
                    if not after['idle']:
                        cell['status']='FAILED_IDLE_AFTER'
                        cell['summary']['matched_decode']={'status':'UNAVAILABLE_UNMATCHED_ISOLATION'}
                        break
            cell['speculative_counter_deltas']={k:after['speculative_counters'][k]-v for k,v in before['speculative_counters'].items() if k in after['speculative_counters']}
            write(a.output/f'cell-{index:03d}.json',{'cell':cell,'requests':rows})
            stopped=cell['status']!='COMPLETED'
        except Exception as e:
            cell.update(status='FAILED_PREFLIGHT_OR_TRANSPORT',error_type=type(e).__name__,
                        error_message=str(e)[:2000])
            stopped=True
        write(a.output/'matrix.json',cells);(a.output/'TABLE.md').write_text(render(cells))
        print(json.dumps(cell),flush=True)
    write(a.output/'matrix.json',cells);(a.output/'TABLE.md').write_text(render(cells))
    if stopped:raise SystemExit(1)


if __name__=='__main__':main()
