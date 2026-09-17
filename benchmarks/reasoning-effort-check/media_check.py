import json, time, urllib.request, hashlib
from pathlib import Path
from vision_acceptance import payload, score, report
root=Path(__file__).parent
fixtures=root/'vision-fixtures'
manifest=json.loads((fixtures/'cases.json').read_text())
output=root/'media-results.json'
assert not output.exists(),'Use a fresh result file'
rows=[]
for case in manifest['cases']:
 with urllib.request.urlopen('http://127.0.0.1:18080/metrics',timeout=15) as response:before=response.read().decode()
 gauges=[l for l in before.splitlines() if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
 assert len(gauges)==2 and all(float(l.rsplit(' ',1)[1])==0 for l in gauges),gauges
 body=payload(case,fixtures,'glm-5.3-flash')
 body.update(max_tokens=256,chat_template_kwargs={'reasoning_effort':'low','enable_thinking':True},response_format={'type':'json_object'})
 started=time.monotonic()
 request=urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=180) as response:result=json.load(response)
 choice=result['choices'][0]
 passed,actual=score(choice['message'].get('content') or '',case['expected'])
 with urllib.request.urlopen('http://127.0.0.1:18080/metrics',timeout=15) as response:after=response.read().decode()
 row={'id':case['id'],'kind':case['kind'],'fixture_sha256':case['sha256'],'passed':passed and choice['finish_reason']=='stop','actual':actual,'expected':case['expected'],'response':result,'seconds':time.monotonic()-started,'metrics_before':before,'metrics_after':after}
 rows.append(row)
 summary=report('glm-5.3-flash',rows,[c['id'] for c in manifest['cases']])
 summary['template']=body['chat_template_kwargs'];summary['max_tokens']=256;summary['response_format']=body['response_format']
 output.write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({k:row[k] for k in ['id','passed','seconds']}),flush=True)
