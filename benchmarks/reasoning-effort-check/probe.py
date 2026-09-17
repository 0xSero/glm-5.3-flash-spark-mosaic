import json, urllib.request, time, sys
base='http://127.0.0.1:18080'
effort=sys.argv[1]
def get(path):
 with urllib.request.urlopen(base+path,timeout=15) as r:return r.read().decode()
before=get('/metrics')
active=[l for l in before.splitlines() if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
assert len(active)==2 and all(float(l.rsplit(' ',1)[1])==0 for l in active),active
payload={'model':'glm-5.3-flash','messages':[{'role':'user','content':'Return only a JSON object with keys answer and numbers. answer must be 6 times 7, and numbers must be the integers from 1 through 20 in order. No other text.'}],'temperature':0,'max_tokens':512,'chat_template_kwargs':{'reasoning_effort':effort,'enable_thinking':True}}
if len(sys.argv)>2 and sys.argv[2]=='json':payload['response_format']={'type':'json_object'}
t=time.monotonic()
req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
elapsed=time.monotonic()-t
choice=result['choices'][0];content=choice['message'].get('content') or ''
try:
 value=json.loads(content);semantic=value=={'answer':42,'numbers':list(range(1,21))}
except (ValueError,TypeError):semantic=False
out={'effort':effort,'payload':payload,'response':result,'elapsed_seconds':elapsed,'semantic_pass':semantic,'accepted':semantic and choice['finish_reason']=='stop','metrics_before':before,'metrics_after':get('/metrics')}
print(json.dumps(out))
