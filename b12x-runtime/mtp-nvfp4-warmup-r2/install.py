import hashlib, importlib.util, json
from pathlib import Path
p=Path(__file__).parent
pins=json.loads((p/'PINS.json').read_text())
target=Path(importlib.util.find_spec('vllm').origin).parent/pins['relative_path']
new=(p/'b12x_warmup.py').read_bytes()
assert hashlib.sha256(new).hexdigest()==pins['after_sha256']
assert hashlib.sha256(target.read_bytes()).hexdigest()==pins['before_sha256']
compile(new,str(target),'exec')
target.write_bytes(new)
print(json.dumps({'state':'INSTALLED_SERVING_UNTESTED',**pins}))
