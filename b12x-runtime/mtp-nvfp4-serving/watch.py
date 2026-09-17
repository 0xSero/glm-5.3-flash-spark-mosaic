"""Readiness, actual graph/resource admission and a bounded useful-output smoke."""
import json,time,traceback,urllib.request
from pathlib import Path
from common import docker,write,sha
import admission
run=Path(__file__).resolve().parent
plan=json.loads((run/'plan.json').read_text())
try:
 for _ in range(1080):
  info=json.loads(docker('inspect',plan['container_name']))[0]
  if not info['State']['Running']:raise RuntimeError('Server exited; retained container/logs')
  try:
   with urllib.request.urlopen('http://127.0.0.1:18080/v1/models',timeout=3) as f:models=json.load(f)
   if [x['id'] for x in models['data']]==['glm-5.3-flash']:break
  except (OSError,ValueError,KeyError):pass
  time.sleep(5)
 else:raise RuntimeError('90 minute readiness deadline; no restart')
 receipt=admission.admit(run,'http://127.0.0.1:18080')
 write(run/'watch-status.json',{'state':'ADMITTED_FUNCTIONAL_SMOKE_PENDING'})
 payload={'model':'glm-5.3-flash','messages':[{'role':'user','content':'Return exactly the JSON object {"answer":42}.'}],'temperature':0,'max_tokens':128,'response_format':{'type':'json_object'},'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'low'}}
 req=urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as f:response=json.load(f)
 write(run/'functional-response.json',{'payload':payload,'response':response})
 choice=response['choices'][0]
 assert choice['finish_reason']=='stop' and json.loads(choice['message']['content'])=={'answer':42}
 write(run/'watch-status.json',{'state':'SERVER_ADMISSION_AND_TEXT_SMOKE_PASSED','admission_sha256':sha(run/'admission.json'),'response_sha256':sha(run/'functional-response.json'),'scope':'Actual capacity and text smoke; full near-limit context, media, MTP accounting, speed and quality remain pending.'})
except BaseException:
 write(run/'watch-status.json',{'state':'FAILED_RETAINED','traceback':traceback.format_exc()})
 raise
