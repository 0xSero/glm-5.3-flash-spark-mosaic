"""Fresh native-MTP output and before/after counter proof; no speed acceptance."""
import json,time,urllib.request,sys
from pathlib import Path
BASE='http://127.0.0.1:18080'
def get(path):
 with urllib.request.urlopen(BASE+path,timeout=10) as r:return r.read().decode()
def metrics():
 return '\n'.join(x for x in get('/metrics').splitlines() if not x.startswith('#') and any(k in x for k in ('spec_decode','gpu_cache_usage','kv_cache_usage','generation_tokens_total','prompt_tokens_total')))
def counter(raw,name):
 return sum(float(line.split()[-1]) for line in raw.splitlines() if line.split('{')[0].split()[0]==name)
def run(prompt):
 before=metrics()
 payload={'model':'glm-5.3-flash','messages':[{'role':'user','content':prompt}],'temperature':0,'stream':True,'return_token_ids':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':True}}
 request=urllib.request.Request(BASE+'/v1/chat/completions',json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 start=time.monotonic();events=[];ttft=None
 with urllib.request.urlopen(request,timeout=300) as response:
  for line in response:
   line=line.decode().strip()
   if not line.startswith('data: '):continue
   raw=line[6:]
   if raw=='[DONE]':break
   event=json.loads(raw);now=time.monotonic()-start
   if ttft is None and any(c.get('token_ids') or c.get('delta',{}).get('content') or c.get('delta',{}).get('reasoning') for c in event.get('choices',[])):ttft=now
   events.append({'time_s':now,'event':event})
 elapsed=time.monotonic()-start;after=metrics()
 content=''.join(c.get('delta',{}).get('content') or '' for row in events for c in row['event'].get('choices',[]))
 reasoning=''.join(c.get('delta',{}).get('reasoning') or c.get('delta',{}).get('reasoning_content') or '' for row in events for c in row['event'].get('choices',[]))
 finish=[c['finish_reason'] for row in events for c in row['event'].get('choices',[]) if c.get('finish_reason')]
 deltas={name:counter(after,name)-counter(before,name) for name in ('vllm:spec_decode_num_draft_tokens_total','vllm:spec_decode_num_accepted_tokens_total')}
 return {'request':payload,'reasoning':reasoning,'content':content,'finish_reasons':finish,'elapsed_s':elapsed,'ttft_s':ttft,'events':events,'metrics_before':before,'metrics_after':after,'mtp_counter_deltas':deltas}
output=Path(sys.argv[1] if len(sys.argv)>1 else 'native-mtp-k256-thinking-smoke.json');assert not output.exists()
records=[]
for prompt in ['What is 6 multiplied by 7? Answer only the number.','Write the integers from 1 through 20, separated by commas. No other text.']:
 records.append(run(prompt));output.write_text(json.dumps({'records':records,'complete':False},indent=2)+'\n')
correct_math=records[0]['content'].strip()=='42'
correct_list=records[1]['content'].replace(' ','').strip().rstrip('.')==','.join(map(str,range(1,21)))
mtp_deltas={name:sum(r['mtp_counter_deltas'][name] for r in records) for name in records[0]['mtp_counter_deltas']}
report={'records':records,'complete':True,'correct_math':correct_math,'correct_list':correct_list,'normal_stops':all(r['finish_reasons']==['stop'] for r in records),'mtp_counter_deltas':mtp_deltas,'pass':correct_math and correct_list and all(r['finish_reasons']==['stop'] for r in records) and all(v>0 for v in mtp_deltas.values())}
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
print(json.dumps([{'content':r['content'],'elapsed_s':r['elapsed_s'],'ttft_s':r['ttft_s']} for r in records],indent=2))
assert report['pass']
