"""One-shot dependent media gate; never overlaps the owned context request."""
import json, os, subprocess, time
from pathlib import Path
ROOT=Path(__file__).parent
PID=1112002
receipt=ROOT/'results-retry/262016.json'
state=ROOT/'after-context-state.json'
assert not state.exists(), 'A watcher already owns this output'
def write(value):
 state.write_text(json.dumps(value,indent=2)+'\n')
write({'phase':'waiting','context_pid':PID,'created':time.time()})
deadline=time.monotonic()+3600
while Path(f'/proc/{PID}').exists():
 cmd=Path(f'/proc/{PID}/cmdline').read_bytes()
 if b'long_context.py' not in cmd:
  raise RuntimeError('Context PID was reused; manual inspection required')
 if time.monotonic()>deadline:
  write({'phase':'timeout','context_pid':PID});raise SystemExit(2)
 time.sleep(15)
if not receipt.exists():
 write({'phase':'context_result_missing','context_pid':PID});raise SystemExit(3)
result=json.loads(receipt.read_text())
if not result.get('accepted') or result.get('prompt_tokens')!=262016:
 write({'phase':'context_failed','context_pid':PID});raise SystemExit(4)
write({'phase':'media_running','context_accepted':True,'time':time.time()})
with (ROOT/'media-client.log').open('x') as log:
 process=subprocess.run(['python3',str(ROOT/'media_check.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
write({'phase':'media_terminal','exit_code':process.returncode,'time':time.time()})
raise SystemExit(process.returncode)
