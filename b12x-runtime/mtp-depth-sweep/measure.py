"""Run the existing timing harness after admission, adding semantic and position gates."""
import argparse,hashlib,importlib.util,json,pathlib,sys
import admission,counters
from common import ROOT,locked_dependencies,sha,write

def validate_task(row,task):
 if not row.get('success') or row.get('finish_reason')!='stop':return False
 v=task['validator'];content=row.get('content','')
 try:
  if v['type']=='json_equals':return json.loads(content)==v['expected']
  if v['type']=='exact_text':return content.strip()==v['expected'].strip()
 except (ValueError,TypeError,KeyError):return False
 raise ValueError('Use a predeclared exact_text or json_equals success contract')

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=pathlib.Path,required=True);parser.add_argument('--task-json',type=pathlib.Path,required=True);parser.add_argument('--prompt-seed',default='glm53-depth-v1')
 a,forward=parser.parse_known_args();forward=forward[1:] if forward[:1]==['--'] else forward
 if not forward:parser.error('Pass existing acceptance/run.py arguments after --')
 locked_dependencies();receipt=admission.verify(a.run);plan=json.loads((a.run/'plan.json').read_text());task_bytes=a.task_json.read_bytes();task=json.loads(task_bytes);task_sha=hashlib.sha256(task_bytes).hexdigest()
 if task.get('content_class') not in ('structured','prose') or not task.get('prompt') or task.get('validator',{}).get('type') not in ('json_equals','exact_text'):raise ValueError('Invalid frozen task contract')
 # Enforce admission-controlled args; users retain ordinary timing/prompt-length knobs.
 controlled={'--model':receipt['model'],'--context':'262144','--max-num-seqs':str(receipt['max_num_seqs']),'--kv-capacity-tokens':str(receipt['kv_capacity_tokens']),'--runtime-receipt':str((a.run/'admission.json').resolve())}
 if any(k in forward or any(x.startswith(k+'=') for x in forward) for k in controlled):raise ValueError('Do not override admission-controlled benchmark arguments')
 if '--plan-only' in forward:raise ValueError('Use launch.py without --execute for offline plans')
 sys.path.insert(0,str(ROOT/'acceptance'));spec=importlib.util.spec_from_file_location('glm_depth_existing_harness',ROOT/'acceptance/run.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 h.TASK='\n'+task['prompt']+'\n';old_stream=h.stream;old_metrics=h.metrics;old_write=h.write;old_prepare=h.prepare;pending=[];prompt_records={};ordinals={}
 def prepare(tokenizer,target,nonce,template):
  ordinal=ordinals.get(target,0);ordinals[target]=ordinal+1
  stable=hashlib.sha256(f'{a.prompt_seed}:{target}:{ordinal}'.encode()).hexdigest()[:32]
  prompt,count=old_prepare(tokenizer,target,stable,template)
  ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,return_dict=False,add_generation_prompt=True,**template)
  if len(ids)!=count:raise ValueError('Frozen prompt token accounting mismatch')
  prompt_records[prompt]={'prompt_token_ids':ids,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'prompt_ordinal':ordinal}
  return prompt,count
 def stream(*args,**kwargs):
  row=old_stream(*args,**kwargs);row.update(prompt_records[args[2]]);row['task_success']=validate_task(row,task)
  if not row['task_success']:row.update(success=False,error_type='DeclaredTaskOrNaturalStopFailure')
  return row
 def metrics(url,model):
  admission.verify(a.run);result=old_metrics(url,model)
  pending.append(counters.snapshot(url,model,plan['depth'],plan['container_name']))
  return result
 def record(path,value):
  if path.name=='run.json':value.update(content_class=task['content_class'],task_name=task.get('name'),task_contract=task,prompt_seed=a.prompt_seed,task_contract_sha256=task_sha,depth_admission_sha256=sha(a.run/'admission.json'))
  if path.name.startswith('cell-') and 'native_timing' in value.get('cell',{}):
   cell=value['cell'];value['position_snapshots']=pending.copy()
   try:
    if len(pending)<2:raise ValueError('Missing bracketing position snapshots')
    cell['position_counter_delta']=counters.compare(pending[0],pending[-1])
   except (ValueError,KeyError) as e:
    cell.update(status='FAILED_POSITION_ACCOUNTING',summary={},native_timing={'status':'FAILED_POSITION_ACCOUNTING'},position_accounting_error=str(e))
   pending.clear()
  old_write(path,value)
 h.stream=stream;h.metrics=metrics;h.write=record;h.prepare=prepare
 sys.argv=[str(ROOT/'acceptance/run.py'),*forward,*[v for pair in controlled.items() for v in pair]]
 h.main()
if __name__=='__main__':main()
