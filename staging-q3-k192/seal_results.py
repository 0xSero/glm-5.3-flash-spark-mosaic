"""CPU-only final Q3 result seal; original results mounted read-only."""
import hashlib,json,sys
from pathlib import Path
candidate=True
r=Path('/release/results')/('massmax-k192');q=Path('/queue')
pin='98e72202d67acc158e908701d04b6d228de74b01d8c1c66d3aa0df6a998b1cbc' if candidate else '05e0ff9cc6a3f87fbd8e27b46bb679e114579dfea3bc4afcc2d724b58be3d1ee'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
report=json.loads((r/'quality-report.json').read_text());identity=json.loads((r/'evaluation-identity.json').read_text())
norm=r/'variant-normalized';m=json.loads((norm/'manifest.json').read_text())
assert report['state']=='QUALITY_MEASURED' and report['tokens']==65504
assert report['candidate_manifest_sha256']==identity['candidate_manifest_sha256']==pin
assert report['experts_per_layer']==192
assert report['fixture_manifest_sha256']==identity['fixture_manifest_sha256']=='46255621cd2da8e05546edc3498cfbee518da092af2f44ae243f79da97c0f00b'
assert report['teacher_manifest_sha256']==identity['teacher_manifest_sha256']=='a1604015f76d6a6b6f032ffb7e3359a770e8dd2564ca89e3b00eb8bcb4d93314'
assert report['variant_normalized_manifest_sha256']==sha(norm/'manifest.json')
assert m['state']=='COMPLETE' and m['rows']==32 and [x['row'] for x in m['records']]==list(range(32))
for item in m['records']:
 p=norm/f"row-{item['row']:03d}.safetensors"
 assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
receipt={'state':'CANDIDATE_Q3_K192_RESULTS_SEALED' if candidate else 'ORIGINAL_Q3_RESULTS_SEALED','report_sha256':sha(r/'quality-report.json'),
         'identity_sha256':sha(r/'evaluation-identity.json'),'normalized_manifest_sha256':sha(norm/'manifest.json'),
         'normalized_rows':32,'scores':report}
(q/('candidate-results-seal.json' if candidate else 'original-q3-results-seal.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'state':receipt['state'],'report_sha256':receipt['report_sha256']}))
