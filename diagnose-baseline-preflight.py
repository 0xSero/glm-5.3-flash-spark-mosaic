import json
import sys
sys.path.insert(0, '/acceptance')
import run
from transformers import AutoTokenizer

print('METRICS', run.metrics('http://127.0.0.1:18080/metrics', 'glm-5.3-flash'), flush=True)
tokenizer = AutoTokenizer.from_pretrained('/model', trust_remote_code=False)
prompt, count = run.prepare(tokenizer, 1024, 'fixed-diagnostic-run', {'enable_thinking': True})
local_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                         add_generation_prompt=True, enable_thinking=True)
check = run.request_json('http://127.0.0.1:18080/tokenize',
    {'model': 'glm-5.3-flash', 'messages': [{'role': 'user', 'content': prompt}],
     'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': True}})
print(json.dumps({'local_count': count, 'server_count': check.get('count'),
                  'server_keys': list(check), 'local_first_ids': local_ids[:12],
                  'server_first_ids': (check.get('tokens') or [])[:12]}), flush=True)
assert check['count'] == count, 'server/local chat tokenization mismatch'
