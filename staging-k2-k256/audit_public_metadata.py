"""Read-only metadata scan. Output is private audit evidence, not a model card."""
import hashlib,json,re,sys
from pathlib import Path
root=Path(sys.argv[1])
pattern=re.compile(r'/Users/|/home/|/workspace/|/mnt/|/root/|/tmp/|/opt/|/var/|/run/|(?:pop-os|popos|omarchy)(?:[.\s/]|$)|spark-(?:2822|2384|557f|de5c)|tailadb2c1|\.ssh/|\b10\.(?:\d{1,3}\.){2}\d{1,3}\b|\b172\.(?:1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3}\b|\b127\.(?:\d{1,3}\.){2}\d{1,3}\b|\b192\.168\.\d{1,3}\.\d{1,3}\b|\b100\.(?:\d{1,3}\.){2}\d{1,3}\b',re.I)
files=[];findings=[]
for path in sorted(root.rglob('*')):
 if not path.is_file() or path.suffix not in ('.json','.txt','.md','.jinja'):continue
 digest=hashlib.sha256();name=str(path.relative_to(root));count=0
 with path.open('rb') as stream:
  for count,line in enumerate(stream,1):
   digest.update(line)
   text=line.decode(errors='replace')
   if pattern.search(text):findings.append({'path':name,'line':count,'excerpt':text.strip()[:220]})
 files.append({'path':name,'bytes':path.stat().st_size,'sha256':digest.hexdigest(),'lines':count})
print(json.dumps({'state':'READ_ONLY_METADATA_SCAN','files':files,'findings':findings,
                 'scope':'Every JSON/text/Markdown/Jinja file including full index and tokenizer; weights not changed or scanned as text.'},indent=2))
