"""Position-preserving, engine-scoped Prometheus snapshots and reset checks."""
import argparse,datetime,json,math,os,re,urllib.request
from common import identity,write
BASES=('vllm:spec_decode_num_drafts','vllm:spec_decode_num_draft_tokens','vllm:spec_decode_num_accepted_tokens','vllm:spec_decode_num_accepted_tokens_per_pos')
LABEL=re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*("(?:[^"\\]|\\.)*")')
LINE=re.compile(r'^([^\s{]+)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$')
def labels(raw):
 raw=raw.strip();out={};pos=0
 while pos<len(raw):
  match=LABEL.match(raw,pos)
  if not match or match[1] in out:raise ValueError('Malformed or duplicate metric label')
  out[match[1]]=json.loads(match[2]);pos=match.end()
  if pos<len(raw):
   if raw[pos]!=',':raise ValueError('Malformed label separator')
   pos+=1
   while pos<len(raw) and raw[pos].isspace():pos+=1
   if pos==len(raw):raise ValueError('Trailing label separator')
 return out

def parse(text,model,depth):
 rows=[];seen=set()
 for line in text.splitlines():
  m=LINE.fullmatch(line)
  if not m or m[1] not in {b+s for b in BASES for s in ('_total','_created')}:continue
  ls=labels(m[2] or '')
  if 'model_name' not in ls:raise ValueError('Speculative metric lacks model identity')
  if ls['model_name']!=model:continue
  if 'engine' not in ls:raise ValueError('Speculative metric lacks engine identity')
  perpos=m[1].startswith(BASES[-1]+'_')
  if perpos:
   if not re.fullmatch(r'0|[1-9][0-9]*',ls.get('position','')) or int(ls['position'])>=depth:raise ValueError('Invalid draft position')
  elif 'position' in ls:raise ValueError('Unexpected position label on aggregate counter')
  value=float(m[3]);key=(m[1],tuple(sorted(ls.items())))
  if key in seen:raise ValueError('Duplicate metric series')
  if not math.isfinite(value) or value<0 or (m[1].endswith('_total') and not value.is_integer()):raise ValueError('Invalid counter value')
  seen.add(key);rows.append({'metric':m[1],'labels':ls,'value':int(value) if m[1].endswith('_total') else value})
 totals=[r for r in rows if r['metric'].endswith('_total')]
 groups={tuple(sorted((k,v) for k,v in r['labels'].items() if k!='position')) for r in totals}
 if not groups:raise ValueError('No matching speculative counters')
 for group in groups:
  got={(r['metric'],r['labels'].get('position')) for r in totals if tuple(sorted((k,v) for k,v in r['labels'].items() if k!='position'))==group}
  expected={(b+'_total',None) for b in BASES[:-1]}|{(BASES[-1]+'_total',str(i)) for i in range(depth)}
  if got!=expected:raise ValueError('Incomplete engine/position counter coverage')
 return sorted(rows,key=lambda r:(r['metric'],tuple(sorted(r['labels'].items()))))

def snapshot(url,model,depth,container):
 before=identity(container);headers={}
 if os.environ.get('MODEL_API_KEY'):headers['Authorization']='Bearer '+os.environ['MODEL_API_KEY']
 with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=15) as response:data=response.read(8*1024*1024+1)
 if len(data)>8*1024*1024:raise ValueError('Oversized metrics response')
 if identity(container)!=before:raise ValueError('Runtime process identity changed during snapshot')
 return {'schema':'glm53-position-counters-v1','model':model,'depth':depth,'identity':before,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'series':parse(data.decode(),model,depth)}

def compare(before,after):
 if any(before[k]!=after[k] for k in ('model','depth','identity')):raise ValueError('Model/depth/process identity changed')
 def index(snapshot):
  result={(r['metric'],tuple(sorted(r['labels'].items()))):r for r in snapshot['series']}
  if len(result)!=len(snapshot['series']):raise ValueError('Duplicate snapshot series')
  return result
 a=index(before);b=index(after)
 if set(a)!=set(b):raise ValueError('Metric label/series set changed')
 rows=[]
 for key in a:
  diff=b[key]['value']-a[key]['value']
  if key[0].endswith('_created'):
   if diff!=0:raise ValueError('Counter creation timestamp changed')
  else:
   if diff<0:raise ValueError('Counter reset detected')
   rows.append({'metric':key[0],'labels':a[key]['labels'],'delta':diff})
 groups={tuple(sorted((k,v) for k,v in r['labels'].items() if k!='position')) for r in rows};engines=[]
 for group in sorted(groups):
  rs=[r for r in rows if tuple(sorted((k,v) for k,v in r['labels'].items() if k!='position'))==group]
  def val(name):return next(r['delta'] for r in rs if r['metric']==name+'_total')
  drafts=val(BASES[0]);drafted=val(BASES[1]);accepted=val(BASES[2]);pos={int(r['labels']['position']):r['delta'] for r in rs if r['metric']==BASES[3]+'_total'}
  if accepted!=sum(pos.values()) or accepted>drafted or drafted>drafts*before['depth'] or any(v>drafts for v in pos.values()):raise ValueError('Inconsistent speculative accounting')
  if any(pos[i]<pos[i+1] for i in range(before['depth']-1)):raise ValueError('Nonmonotonic accepted-prefix positions')
  engines.append({'labels':dict(group),'drafts':drafts,'drafted_tokens':drafted,'accepted_tokens':accepted,'accepted_tokens_by_position':pos,'acceptance_rate_by_position':{i:v/drafts if drafts else None for i,v in pos.items()}})
 drafts=sum(e['drafts'] for e in engines);drafted=sum(e['drafted_tokens'] for e in engines);accepted=sum(e['accepted_tokens'] for e in engines)
 if drafts<=0 or drafted<=0:raise ValueError('Speculative counters did not advance')
 return {'state':'MATCHED_POSITION_COUNTERS','engines':engines,'series_deltas':rows,'drafts':drafts,'drafted_tokens':drafted,'accepted_tokens':accepted,'accepted_fraction':accepted/drafted,'mean_acceptance_length':1+accepted/drafts}

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 s=sub.add_parser('snapshot');s.add_argument('--url',required=True);s.add_argument('--model',required=True);s.add_argument('--depth',type=int,choices=(1,2,3,5),required=True);s.add_argument('--container',required=True)
 d=sub.add_parser('diff');d.add_argument('--before',required=True);d.add_argument('--after',required=True)
 for command in (s,d):command.add_argument('--output',required=True)
 a=p.parse_args()
 if os.path.exists(a.output):p.error('Use a fresh output path')
 result=snapshot(a.url,a.model,a.depth,a.container) if a.command=='snapshot' else compare(json.load(open(a.before)),json.load(open(a.after)))
 write(a.output,result)
if __name__=='__main__':main()
