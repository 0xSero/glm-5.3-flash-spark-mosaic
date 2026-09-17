"""Bounded readiness -> actual admission -> functional smoke -> exact replay; no restart."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'b12x-runtime/mtp-depth-sweep'))
import admission
from common import docker, sha, write

p=argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
p.add_argument('--fixture', type=Path, required=True)
a=p.parse_args()
plan=json.loads((a.run/'plan.json').read_text())
write(a.run/'watcher-source.json', {'sha256':sha(__file__),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_gpu_restart':True})
try:
    deadline=time.monotonic()+5400
    while time.monotonic()<deadline:
        info=json.loads(docker('inspect',plan['container_name']))[0]
        if not info['State']['Running']:
            raise RuntimeError('Runtime exited before readiness; retain container and logs')
        try:
            with urllib.request.urlopen('http://127.0.0.1:18080/v1/models',timeout=3) as f: models=json.load(f)
            if [m['id'] for m in models['data']]==['glm-5.3-flash']:
                break
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(5)
    else:
        raise RuntimeError('Readiness deadline exceeded; no process changes')
    receipt=admission.admit(a.run,'http://127.0.0.1:18080')
    write(a.run/'watcher-status.json',{'phase':'admitted_functional_smoke_pending'})
    payload={'model':'glm-5.3-flash','messages':[{'role':'user','content':'Return exactly the JSON object {"answer":42}.'}],
        'temperature':0,'max_tokens':128,'response_format':{'type':'json_object'},
        'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'low'}}
    req=urllib.request.Request('http://127.0.0.1:18080/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as f: response=json.load(f)
    write(a.run/'initialization-response.json',{'payload':payload,'response':response})
    c=response['choices'][0]
    if c['finish_reason']!='stop' or json.loads(c['message']['content'])!={'answer':42}:
        raise ValueError('Functional smoke failed')
    write(a.run/'watcher-status.json',{'phase':'exact_prompt_replay_running'})
    out=a.run/'exact-baseline-replay'
    subprocess.run([sys.executable,str(ROOT/'b12x-runtime/mtp-baseline-replay/replay.py'),'--run',str(a.run),'--fixture',str(a.fixture),'--output',str(out),'--targets','1024,16384'],check=True)
    with (a.run/'verification.log').open('x') as f:
        subprocess.run([sys.executable,str(ROOT/'benchmarks/verify_structured.py'),str(out),'--last-integer','300'],stdout=f,stderr=subprocess.STDOUT,check=True)
    write(a.run/'watcher-status.json',{'phase':'completed_results_require_review','verified_sha256':sha(out/'verified.json'),'scope':'Shorter than30s results remain screens, not sustained throughput'})
except BaseException:
    write(a.run/'watcher-status.json',{'phase':'failed','traceback':traceback.format_exc(),'no_gpu_restart':True})
    raise
