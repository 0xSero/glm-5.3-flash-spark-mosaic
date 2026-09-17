import json,urllib.request,time,re
from pathlib import Path
out=Path('/home/sero/work/glm53-single-spark-release-20260911/b12x-runtime/semantic-smoke');out.mkdir(exist_ok=True)
base='http://127.0.0.1:18080'
def get(path):return urllib.request.urlopen(base+path,timeout=10).read()
(out/'metrics-before.txt').write_bytes(get('/metrics'))
rows=[]
for name,prompt in [('arithmetic','What is 6 multiplied by 7? Reply with only the final integer.'),('count','Count from 1 through 20, inclusive, in increasing order. Output only the numbers, separated by commas.')]:
 payload={'model':'glm-5.3-flash','messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':256,'chat_template_kwargs':{'enable_thinking':True}}
 req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 start=time.perf_counter()
 with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
 elapsed=time.perf_counter()-start
 (out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 choice=result['choices'][0];content=choice['message'].get('content') or ''
 nums=[int(x) for x in re.findall(r'\d+',content)]
 passed=nums==([42] if name=='arithmetic' else list(range(1,21))) and choice['finish_reason']=='stop'
 rows.append({'name':name,'passed':passed,'content':content,'finish_reason':choice['finish_reason'],'usage':result.get('usage'),'elapsed_seconds':elapsed,'reasoning_present':bool(choice['message'].get('reasoning') or choice['message'].get('reasoning_content'))})
 print(json.dumps(rows[-1]),flush=True)
(out/'metrics-after.txt').write_bytes(get('/metrics'))
report={'state':'FRESH_SEMANTIC_PASS' if all(r['passed'] for r in rows) else 'SEMANTIC_FAILURE','requests':rows,'not_a_throughput_benchmark':True}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
assert all(r['passed'] for r in rows),report
