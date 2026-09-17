"""Package private arrays with immutable source and arithmetic identity."""
import hashlib,json
from pathlib import Path
w=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
i=json.loads((w/'identity.json').read_text())
for name,key in [('input_ids.json','input_ids_json_sha256'),('teacher_top1.json','teacher_top1_json_sha256')]:assert sha(w/name)==i[key]
r={**i,'input_ids':json.loads((w/'input_ids.json').read_text()),'teacher_top1':json.loads((w/'teacher_top1.json').read_text()),'head_shard_sha256':i['native_head_shard_sha256'],'teacher_nll_per_row':i['teacher_nll_sums'],'identity_sha256':sha(w/'identity.json')}
assert len(r['input_ids'])==len(r['teacher_top1'])==32
assert all(len(x)==2048 for x in r['input_ids']) and all(len(x)==2047 for x in r['teacher_top1'])
with (w/'reference.json').open('x') as f:json.dump(r,f,separators=(',',':'),allow_nan=False);f.write('\n')
(w/'reference.sha256').write_text(sha(w/'reference.json')+'  reference.json\n')
print(sha(w/'reference.json'))
