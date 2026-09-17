"""Check actual prompt-logprob API support before a serving-quality evaluation."""
import json,math,time,urllib.request
from pathlib import Path

p=json.loads(Path('/acceptance/original-k2-runtime-parity-probes.json').read_text())['probes'][0]
ids=p['prefix_ids']+[p['label_token_id']]
body={'model':'glm-5.3-flash','prompt':ids,'max_tokens':1,'temperature':0,
      'echo':True,'logprobs':1,'return_tokens_as_token_ids':True,
      'return_token_ids':True,'add_special_tokens':False}
request=urllib.request.Request('http://127.0.0.1:18080/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
t=time.monotonic()
with urllib.request.urlopen(request,timeout=180) as response:data=json.load(response)
out={'scope':'Prompt-logprob contract probe, not a full quality result','row':0,'seconds':time.monotonic()-t,'response':data}
Path('/results/k2full-fp8-prompt-logprobs-contract-r4.json').write_text(json.dumps(out,indent=2)+'\n')
logs=data['choices'][0].get('logprobs') or {}
print(json.dumps({'usage':data.get('usage'),'arrays':{k:len(v) for k,v in logs.items() if isinstance(v,list)},
      'first_logprob':(logs.get('token_logprobs') or [None])[0],
      'first_token':(logs.get('tokens') or [None])[0],
      'second_top_logprobs':(logs.get('top_logprobs') or [{},{}])[1]}))
